#!/usr/bin/env python
"""
Documentation Reference Validator

This tool recursively analyzes the codebase to ensure all referenced documentation 
exists and accurately reflects the actual code implementation. It builds a dependency 
graph of referenced documentation files and validates that documentation is complete 
and accurate relative to the code it documents.

Usage:
    python -m app.tools.documentation.doc_reference_validator --index-file /path/to/index.md

Features:
- Verify physical existence of referenced documentation
- Confirm content aligns with actual code implementation
- Check for documentation completeness relative to code complexity
- Validate that architectural descriptions match the actual code organization
- Generate placeholders for missing documentation

"""
import argparse
import ast
import glob
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union

import networkx as nx
import markdown
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("doc_reference_validator")

# Add this constant at the top of the file, after the imports
MAX_PATH_LENGTH = 1000  # Maximum safe path length in bytes

class DocumentationType(Enum):
    """Types of documentation references."""
    MARKDOWN = "markdown"
    PYTHON_DOCSTRING = "docstring"
    CODE_COMMENT = "comment"
    RST = "rst"
    UNKNOWN = "unknown"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DocumentationReference:
    """Represents a reference to a documentation file."""
    source_file: Path
    target_file: Path
    line_number: int
    reference_type: DocumentationType
    reference_text: str
    context: str = ""


@dataclass
class ValidationIssue:
    """Represents an issue found during validation."""
    file_path: Path
    line_number: int
    message: str
    severity: ValidationSeverity
    suggested_fix: Optional[str] = None
    reference: Optional[DocumentationReference] = None


@dataclass
class CodeStructure:
    """Represents a structure in the code (class, function, etc.)."""
    name: str
    file_path: Path
    line_start: int
    line_end: int
    type: str  # 'class', 'function', 'method', etc.
    docstring: Optional[str] = None
    signature: Optional[str] = None
    parent: Optional['CodeStructure'] = None
    children: List['CodeStructure'] = field(default_factory=list)
    complexity: int = 0
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None


@dataclass
class ValidationReport:
    """Report generated after validation."""
    issues: List[ValidationIssue] = field(default_factory=list)
    references: List[DocumentationReference] = field(default_factory=list)
    missing_documentation: List[Path] = field(default_factory=list)
    incomplete_documentation: List[Tuple[Path, str]] = field(default_factory=list)
    code_structure: Dict[Path, List[CodeStructure]] = field(default_factory=dict)
    
    @property
    def error_count(self) -> int:
        """Count of error and critical issues."""
        return sum(1 for issue in self.issues 
                  if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
    
    @property
    def warning_count(self) -> int:
        """Count of warning issues."""
        return sum(1 for issue in self.issues 
                  if issue.severity == ValidationSeverity.WARNING)
    
    def to_json(self) -> str:
        """Convert report to JSON format."""
        return json.dumps({
            "issues": [
                {
                    "file": str(issue.file_path),
                    "line": issue.line_number,
                    "message": issue.message,
                    "severity": issue.severity.value,
                    "suggested_fix": issue.suggested_fix
                }
                for issue in self.issues
            ],
            "missing_documentation": [str(path) for path in self.missing_documentation],
            "incomplete_documentation": [
                {"file": str(path), "details": details}
                for path, details in self.incomplete_documentation
            ],
            "summary": {
                "total_issues": len(self.issues),
                "errors": self.error_count,
                "warnings": self.warning_count,
                "references_analyzed": len(self.references)
            }
        }, indent=2)
    
    def to_markdown(self) -> str:
        """Convert report to Markdown format."""
        md = []
        md.append("# Documentation Reference Validation Report\n")
        
        md.append("## Summary\n")
        md.append(f"- Total issues: {len(self.issues)}")
        md.append(f"- Errors: {self.error_count}")
        md.append(f"- Warnings: {self.warning_count}")
        md.append(f"- References analyzed: {len(self.references)}")
        md.append(f"- Missing documentation files: {len(self.missing_documentation)}")
        md.append(f"- Incomplete documentation: {len(self.incomplete_documentation)}")
        md.append("\n")
        
        if self.issues:
            md.append("## Issues\n")
            for issue in sorted(self.issues, key=lambda x: (x.severity.value, x.file_path, x.line_number)):
                md.append(f"### {issue.severity.value.upper()}: {issue.message}")
                md.append(f"- File: `{issue.file_path}`")
                md.append(f"- Line: {issue.line_number}")
                if issue.suggested_fix:
                    md.append("- Suggested fix:")
                    md.append(f"```\n{issue.suggested_fix}\n```")
                md.append("\n")
        
        if self.missing_documentation:
            md.append("## Missing Documentation Files\n")
            for path in sorted(self.missing_documentation):
                md.append(f"- `{path}`")
            md.append("\n")
        
        if self.incomplete_documentation:
            md.append("## Incomplete Documentation\n")
            for path, details in sorted(self.incomplete_documentation, key=lambda x: x[0]):
                md.append(f"### `{path}`")
                md.append(details)
                md.append("\n")
        
        return "\n".join(md)


class DocReferenceValidator:
    """
    Validates documentation references in a codebase.
    
    This class recursively analyzes the codebase starting from a specified index file,
    builds a graph of documentation references, and validates that all references
    point to existing documentation that accurately reflects the code.
    """
    
    def __init__(
        self, 
        index_file: str,
        workspace_root: Optional[str] = None,
        exclude_dirs: Optional[List[str]] = None,
        exclude_files: Optional[List[str]] = None
    ):
        """
        Initialize the validator.
        
        Args:
            index_file: Path to the starting index file
            workspace_root: Root directory of the workspace
            exclude_dirs: Directories to exclude from analysis
            exclude_files: Files to exclude from analysis
        """
        self.index_file = Path(index_file)
        
        if workspace_root:
            self.workspace_root = Path(workspace_root)
        else:
            # Try to determine workspace root
            if os.environ.get("WORKSPACE_ROOT"):
                self.workspace_root = Path(os.environ["WORKSPACE_ROOT"])
            else:
                # Find git root
                current = self.index_file.parent
                while current != current.parent:
                    if (current / ".git").exists():
                        self.workspace_root = current
                        break
                    current = current.parent
                else:
                    # Fallback to index file parent
                    self.workspace_root = self.index_file.parent
        
        self.exclude_dirs = exclude_dirs or [
            ".git", "__pycache__", "venv", "env", ".env", "node_modules"
        ]
        self.exclude_files = exclude_files or []
        
        self.references: List[DocumentationReference] = []
        self.reference_graph = nx.DiGraph()
        self.code_structures: Dict[Path, List[CodeStructure]] = {}
        self.doc_content_cache: Dict[Path, str] = {}
        self.markdown_link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
        self.markdown_ref_pattern = re.compile(r"<([^>]+)>|\[([^\]]+)\]\[([^\]]+)\]")
        self.python_reference_pattern = re.compile(
            r"(?:See|see|reference|Reference|doc|documentation|docs|Documentation):\s*`?([^`\n]+)`?"
        )
        
        # Patterns for file extensions
        self.code_extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp"}
        self.doc_extensions = {".md", ".rst", ".txt", ".html", ".ipynb"}
        
    def validate(self) -> ValidationReport:
        """
        Perform the validation and return a report.
        
        Returns:
            ValidationReport: The validation report
        """
        logger.info(f"Starting validation from index file: {self.index_file}")
        logger.info(f"Workspace root: {self.workspace_root}")
        
        # Initialize report
        report = ValidationReport()
        
        # Build code structure
        self._analyze_code_structure()
        report.code_structure = self.code_structures
        
        # Extract references from the index file
        self._extract_references_from_file(self.index_file)
        
        # Create reference graph
        self._build_reference_graph()
        
        # Analyze referenced files recursively
        analyzed_files = set()
        files_to_analyze = {ref.target_file for ref in self.references if ref.target_file is not None}
        
        while files_to_analyze:
            current_file = files_to_analyze.pop()
            if current_file in analyzed_files:
                continue
            
            analyzed_files.add(current_file)
            
            # Skip files with excessively long paths
            if len(str(current_file).encode('utf-8')) > MAX_PATH_LENGTH:
                logger.warning(f"Skipping file with excessively long path: {str(current_file)[:100]}...")
                issue = ValidationIssue(
                    file_path=Path(str(current_file)[:100] + "..."),
                    line_number=0,
                    message=f"File path is too long for processing",
                    severity=ValidationSeverity.WARNING
                )
                report.issues.append(issue)
                continue
            
            if current_file.exists():
                try:
                    new_refs = self._extract_references_from_file(current_file)
                    # Only add valid target files to the list
                    files_to_analyze.update({ref.target_file for ref in new_refs if ref.target_file is not None})
                except OSError as e:
                    logger.warning(f"Error processing file {current_file}: {e}")
                    issue = ValidationIssue(
                        file_path=current_file,
                        line_number=0,
                        message=f"Error processing file: {e}",
                        severity=ValidationSeverity.WARNING
                    )
                    report.issues.append(issue)
            else:
                issue = ValidationIssue(
                    file_path=current_file,
                    line_number=0,
                    message=f"Referenced documentation file does not exist",
                    severity=ValidationSeverity.ERROR,
                    suggested_fix=f"Create the missing file at {current_file}"
                )
                report.issues.append(issue)
                report.missing_documentation.append(current_file)
        
        # Validate all references
        for ref in self.references:
            # Skip references with invalid target files
            if ref.target_file is None:
                continue
            
            issues = self._validate_reference(ref)
            report.issues.extend(issues)
        
        # Check for code without documentation
        for file_path, structures in self.code_structures.items():
            for structure in structures:
                if self._needs_documentation(structure) and not structure.docstring:
                    issue = ValidationIssue(
                        file_path=file_path,
                        line_number=structure.line_start,
                        message=f"Missing docstring for {structure.type} '{structure.name}'",
                        severity=ValidationSeverity.WARNING,
                        suggested_fix=self._generate_docstring(structure)
                    )
                    report.issues.append(issue)
        
        # Add references to report
        report.references = self.references
        
        logger.info(f"Validation complete. Found {len(report.issues)} issues.")
        return report

    def _analyze_code_structure(self) -> None:
        """Analyze the structure of code files in the workspace."""
        logger.info("Analyzing code structure...")
        
        # Find all Python files
        python_files = []
        for root, dirs, files in os.walk(self.workspace_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            for file in files:
                if file.endswith(".py") and file not in self.exclude_files:
                    full_path = Path(os.path.join(root, file))
                    python_files.append(full_path)
        
        # Parse each file
        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                tree = ast.parse(content)
                structures = self._extract_structures_from_ast(tree, file_path)
                self.code_structures[file_path] = structures
            except Exception as e:
                logger.warning(f"Error parsing {file_path}: {e}")
    
    def _extract_structures_from_ast(self, tree: ast.AST, file_path: Path) -> List[CodeStructure]:
        """Extract code structures from an AST."""
        structures = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node)
                
                if isinstance(node, ast.ClassDef):
                    structure_type = "class"
                    signature = f"class {node.name}"
                    params = []
                    return_type = None
                else:
                    structure_type = "function" if isinstance(node, ast.FunctionDef) else "async function"
                    params = []
                    return_type = None
                    
                    # Extract parameters
                    for arg in node.args.args:
                        param = {"name": arg.arg}
                        if arg.annotation:
                            param["type"] = ast.unparse(arg.annotation)
                        params.append(param)
                    
                    # Extract return type
                    if node.returns:
                        return_type = ast.unparse(node.returns)
                    
                    # Build signature
                    param_strs = []
                    for param in params:
                        param_str = param["name"]
                        if param.get("type"):
                            param_str += f": {param['type']}"
                        param_strs.append(param_str)
                    
                    signature = f"def {node.name}({', '.join(param_strs)})"
                    if return_type:
                        signature += f" -> {return_type}"
                
                # Calculate complexity using cyclomatic complexity
                complexity = 1  # Base complexity
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler)):
                        complexity += 1
                    elif isinstance(child, ast.BoolOp) and isinstance(child.op, (ast.And, ast.Or)):
                        complexity += len(child.values) - 1
                
                structure = CodeStructure(
                    name=node.name,
                    file_path=file_path,
                    line_start=node.lineno,
                    line_end=node.end_lineno,
                    type=structure_type,
                    docstring=docstring,
                    signature=signature,
                    complexity=complexity,
                    parameters=params,
                    return_type=return_type
                )
                
                structures.append(structure)
        
        return structures
    
    def _extract_references_from_file(self, file_path: Path) -> List[DocumentationReference]:
        """
        Extract documentation references from a file.
        
        Args:
            file_path: Path to the file to extract references from
            
        Returns:
            List of DocumentationReference objects
        """
        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return []
        
        if file_path.suffix in self.doc_extensions:
            return self._extract_references_from_doc_file(file_path)
        elif file_path.suffix in self.code_extensions:
            return self._extract_references_from_code_file(file_path)
        else:
            # Skip binary files and other non-text files
            return []
    
    def _extract_references_from_doc_file(self, file_path: Path) -> List[DocumentationReference]:
        """Extract references from a documentation file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.doc_content_cache[file_path] = content
        except UnicodeDecodeError:
            logger.warning(f"Cannot decode file as text: {file_path}")
            return []
        except OSError as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return []
        
        references = []
        
        # Extract Markdown links
        for match in self.markdown_link_pattern.finditer(content):
            try:
                link_text, target = match.groups()
                line_number = content[:match.start()].count('\n') + 1
                
                # Normalize the target path
                target_path = self._normalize_path(file_path, target)
                if target_path:
                    # Skip paths that are too long
                    if len(str(target_path).encode('utf-8')) > MAX_PATH_LENGTH:
                        logger.warning(f"Skipping reference with excessively long path: {str(target_path)[:100]}...")
                        continue
                        
                    reference = DocumentationReference(
                        source_file=file_path,
                        target_file=target_path,
                        line_number=line_number,
                        reference_type=DocumentationType.MARKDOWN,
                        reference_text=target,
                        context=content[max(0, match.start() - 50):match.end() + 50]
                    )
                    references.append(reference)
                    self.references.append(reference)
            except Exception as e:
                # Catch any errors during reference extraction and continue
                # Calculate line number separately to avoid f-string issues
                line_num = content[:match.start()].count('\n') + 1
                logger.warning(f"Error extracting reference from {file_path} at line {line_num}: {e}")
                continue
        
        return references
    
    def _extract_references_from_code_file(self, file_path: Path) -> List[DocumentationReference]:
        """Extract references from a code file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            logger.warning(f"Cannot decode file as text: {file_path}")
            return []
        except OSError as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return []
        
        references = []
        
        # Extract references from Python docstrings and comments
        lines = content.split('\n')
        for i, line in enumerate(lines):
            try:
                # Look for references in comments
                match = self.python_reference_pattern.search(line)
                if match:
                    target = match.group(1).strip()
                    target_path = self._normalize_path(file_path, target)
                    if target_path:
                        # Skip paths that are too long
                        if len(str(target_path).encode('utf-8')) > MAX_PATH_LENGTH:
                            logger.warning(f"Skipping reference with excessively long path: {str(target_path)[:100]}...")
                            continue
                            
                        reference = DocumentationReference(
                            source_file=file_path,
                            target_file=target_path,
                            line_number=i + 1,
                            reference_type=DocumentationType.CODE_COMMENT,
                            reference_text=target,
                            context='\n'.join(lines[max(0, i - 2):min(len(lines), i + 3)])
                        )
                        references.append(reference)
                        self.references.append(reference)
            except Exception as e:
                # Catch any errors during reference extraction and continue
                logger.warning(f"Error extracting reference from {file_path} at line {i + 1}: {e}")
                continue
        
        # Parse Python docstrings for references using AST
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        try:
                            # Look for references in docstring
                            for match in self.python_reference_pattern.finditer(docstring):
                                target = match.group(1).strip()
                                target_path = self._normalize_path(file_path, target)
                                if target_path:
                                    # Skip paths that are too long
                                    if len(str(target_path).encode('utf-8')) > MAX_PATH_LENGTH:
                                        logger.warning(f"Skipping reference with excessively long path: {str(target_path)[:100]}...")
                                        continue
                                        
                                    reference = DocumentationReference(
                                        source_file=file_path,
                                        target_file=target_path,
                                        line_number=node.lineno,
                                        reference_type=DocumentationType.PYTHON_DOCSTRING,
                                        reference_text=target,
                                        context=docstring[:200] + ("..." if len(docstring) > 200 else "")
                                    )
                                    references.append(reference)
                                    self.references.append(reference)
                        except Exception as e:
                            # Catch any errors during docstring reference extraction
                            logger.warning(f"Error extracting docstring reference from {file_path} at line {node.lineno}: {e}")
                            continue
        except SyntaxError:
            # Not a valid Python file or syntax error
            pass
        except Exception as e:
            logger.warning(f"Error parsing AST for {file_path}: {e}")
        
        return references
    
    def _normalize_path(self, source_file: Union[Path, str], target_reference: Optional[str] = None) -> Optional[Path]:
        """
        Normalize a reference path.
        
        Args:
            source_file: Path of the file containing the reference, or the path to normalize if target_reference is None
            target_reference: The reference string or None if source_file is the path to normalize
            
        Returns:
            Normalized target path or None if the reference doesn't point to a file
        """
        # If target_reference is None, then source_file is the path to normalize
        if target_reference is None:
            path_to_normalize = source_file
            # Use workspace root as source directory
            source_dir = self.workspace_root
        else:
            path_to_normalize = target_reference
            # Use source file's directory as the reference point
            source_dir = Path(source_file).parent if isinstance(source_file, (str, Path)) else self.workspace_root

        # Skip links that are URLs
        if isinstance(path_to_normalize, str) and path_to_normalize.startswith(("http://", "https://", "ftp://")):
            return None
        
        # Skip anchors without a file reference
        if isinstance(path_to_normalize, str) and path_to_normalize.startswith('#'):
            return None
        
        # Handle anchor in file reference
        if isinstance(path_to_normalize, str) and '#' in path_to_normalize:
            path_to_normalize = path_to_normalize.split('#')[0]
        
        # Handle empty references
        if not path_to_normalize:
            return None
        
        try:
            # Convert to string if it's a Path object
            if isinstance(path_to_normalize, Path):
                path_to_normalize = str(path_to_normalize)
            
            # Handle absolute paths
            if path_to_normalize.startswith('/'):
                return (self.workspace_root / path_to_normalize.lstrip('/')).resolve()
            
            # Handle relative paths - use resolve() to normalize paths with many '../' sequences
            relative_path = source_dir / path_to_normalize
            return relative_path.resolve()
        except (ValueError, RuntimeError) as e:
            # Handle path resolution errors (like too many symlinks or circular references)
            logger.warning(f"Error normalizing path '{path_to_normalize}' relative to '{source_dir}': {e}")
            return None
    
    def _build_reference_graph(self) -> None:
        """Build a directed graph of documentation references."""
        for ref in self.references:
            if ref.target_file is not None:
                self.reference_graph.add_edge(ref.source_file, ref.target_file)
    
    def _validate_reference(self, ref: DocumentationReference) -> List[ValidationIssue]:
        """
        Validate a documentation reference.
        
        Args:
            ref: The reference to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check if target file exists
        if not ref.target_file.exists():
            issue = ValidationIssue(
                file_path=ref.source_file,
                line_number=ref.line_number,
                message=f"Reference points to non-existent file: {ref.target_file}",
                severity=ValidationSeverity.ERROR,
                reference=ref,
                suggested_fix=f"Create the missing file at {ref.target_file}"
            )
            issues.append(issue)
            return issues
        
        # Check if target is a documentation file
        if ref.target_file.suffix not in self.doc_extensions:
            # Not necessarily an issue, could be a code reference
            return issues
        
        # Check if target file contains content about the source code
        if ref.source_file.suffix in self.code_extensions:
            # This is a code file referencing documentation
            # Get the code structures for the source file
            structures = self.code_structures.get(ref.source_file, [])
            if not structures:
                # No code structures found for this file
                return issues
            
            # Load target content if not cached
            if ref.target_file not in self.doc_content_cache:
                try:
                    with open(ref.target_file, "r", encoding="utf-8") as f:
                        self.doc_content_cache[ref.target_file] = f.read()
                except Exception as e:
                    issue = ValidationIssue(
                        file_path=ref.target_file,
                        line_number=0,
                        message=f"Error reading documentation file: {e}",
                        severity=ValidationSeverity.WARNING,
                        reference=ref
                    )
                    issues.append(issue)
                    return issues
            
            content = self.doc_content_cache[ref.target_file]
            
            # Check if documentation mentions class/function names
            for structure in structures:
                if structure.name not in content:
                    issue = ValidationIssue(
                        file_path=ref.target_file,
                        line_number=0,
                        message=f"Documentation does not mention {structure.type} '{structure.name}' from {ref.source_file}",
                        severity=ValidationSeverity.WARNING,
                        reference=ref,
                        suggested_fix=f"Add documentation for {structure.type} '{structure.name}'"
                    )
                    issues.append(issue)
                
                # Check for parameters documentation
                if hasattr(structure, 'parameters') and structure.parameters:
                    for param in structure.parameters:
                        param_name = param["name"]
                        # Skip self, cls in methods
                        if param_name in ("self", "cls"):
                            continue
                        if param_name not in content:
                            issue = ValidationIssue(
                                file_path=ref.target_file,
                                line_number=0,
                                message=f"Documentation does not mention parameter '{param_name}' of {structure.type} '{structure.name}'",
                                severity=ValidationSeverity.INFO,
                                reference=ref,
                                suggested_fix=f"Add documentation for parameter '{param_name}'"
                            )
                            issues.append(issue)
                            
                # Check for return type documentation
                if structure.return_type and "return" not in content.lower():
                    issue = ValidationIssue(
                        file_path=ref.target_file,
                        line_number=0,
                        message=f"Documentation does not mention return value of {structure.type} '{structure.name}'",
                        severity=ValidationSeverity.INFO,
                        reference=ref,
                        suggested_fix=f"Add documentation for return value of type '{structure.return_type}'"
                    )
                    issues.append(issue)
        
        return issues
    
    def _needs_documentation(self, structure: CodeStructure) -> bool:
        """Determine if a code structure needs documentation."""
        # Skip private members (starting with underscore)
        if structure.name.startswith('_') and not structure.name.startswith('__'):
            return False
        
        # Always document classes
        if structure.type == "class":
            return True
        
        # Document complex functions
        if structure.complexity > 3:
            return True
        
        # Document functions with parameters
        if structure.parameters and any(p["name"] not in ("self", "cls") for p in structure.parameters):
            return True
        
        # Document functions with return values
        if structure.return_type and structure.return_type != "None":
            return True
        
        return False
    
    def _generate_docstring(self, structure: CodeStructure) -> str:
        """Generate a docstring template for a code structure."""
        if structure.type == "class":
            docstring = f'"""\n{structure.name}\n\n'
            docstring += "TODO: Add class description\n"
            docstring += '"""'
            return docstring
        
        docstring = f'"""\n'
        docstring += "TODO: Add function description\n\n"
        
        if structure.parameters:
            docstring += "Args:\n"
            for param in structure.parameters:
                if param["name"] in ("self", "cls"):
                    continue
                param_type = f" ({param.get('type', 'Any')})" if param.get('type') else ""
                docstring += f"    {param['name']}{param_type}: TODO: Add description\n"
        
        if structure.return_type and structure.return_type != "None":
            docstring += "\nReturns:\n"
            docstring += f"    {structure.return_type}: TODO: Add description\n"
        
        docstring += '"""'
        return docstring

    def _load_doc_tracker(self, doc_tracker_path: str) -> Dict[str, bool]:
        """
        Load the documentation tracker file and extract high-priority document paths.
        
        Args:
            doc_tracker_path: Path to the documentation tracker markdown file
            
        Returns:
            A dictionary mapping file paths to their importance (True if high priority)
        """
        if not os.path.exists(doc_tracker_path):
            logger.warning(f"Documentation tracker file not found: {doc_tracker_path}")
            return {}
            
        try:
            with open(doc_tracker_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse markdown table to extract file paths and their status
            # Look for table rows with file paths and status indicators
            high_priority_files = {}
            
            # Extract table rows with regex
            table_pattern = r'\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|'
            matches = re.finditer(table_pattern, content)
            
            for match in matches:
                file_path = match.group(1).strip()
                status = match.group(2).strip()
                
                # Skip header rows and empty rows
                if file_path == "Document" or file_path == "---" or file_path == "":
                    continue
                    
                # Consider it high priority if it's marked as missing (❌)
                is_high_priority = "❌" in status
                
                # Clean up file path - extract from markdown links if present
                link_match = re.search(r'\[(.*?)\]\((.*?)\)', file_path)
                if link_match:
                    file_path = link_match.group(2).strip()
                
                # Normalize the path
                normalized_path = self._normalize_path(file_path)
                if normalized_path:
                    high_priority_files[normalized_path] = is_high_priority
            
            logger.info(f"Loaded {len(high_priority_files)} documents from tracker")
            return high_priority_files
            
        except Exception as e:
            logger.warning(f"Error parsing documentation tracker: {e}")
            return {}
    
    def is_high_priority(self, file_path: str, high_priority_files: Dict[str, bool]) -> bool:
        """
        Determine if a file is high priority based on the documentation tracker.
        
        Args:
            file_path: Path to the file
            high_priority_files: Dictionary of high priority files
            
        Returns:
            True if the file is high priority, False otherwise
        """
        normalized_path = self._normalize_path(file_path)
        if not normalized_path:
            return False
            
        # Check if the file is in the high priority list
        return high_priority_files.get(normalized_path, False)
    
    def filter_high_priority_issues(self, report: ValidationReport, 
                                  high_priority_files: Dict[str, bool]) -> ValidationReport:
        """
        Filter the validation report to include only high-priority issues.
        
        Args:
            report: The validation report to filter
            high_priority_files: Dictionary of high priority files
            
        Returns:
            A filtered validation report
        """
        filtered_report = ValidationReport(
            issues=[],
            references=report.references,
            missing_documentation=[],
            incomplete_documentation=[]
        )
        
        # Filter missing documentation to include only high-priority files
        filtered_report.missing_documentation = [
            doc for doc in report.missing_documentation
            if self.is_high_priority(str(doc), high_priority_files)
        ]
        
        # Filter issues to include only those affecting high-priority files
        filtered_report.issues = [
            issue for issue in report.issues
            if self.is_high_priority(str(issue.file_path), high_priority_files)
        ]
        
        # Filter incomplete documentation
        filtered_report.incomplete_documentation = [
            (path, details) for path, details in report.incomplete_documentation
            if self.is_high_priority(str(path), high_priority_files)
        ]
        
        # Copy code structure information
        filtered_report.code_structure = report.code_structure
        
        return filtered_report


def main():
    """Main entry point for the validator."""
    parser = argparse.ArgumentParser(
        description="Validate documentation references in a codebase."
    )
    parser.add_argument(
        "--index-file", 
        required=True,
        help="Path to the index file to start validation from"
    )
    parser.add_argument(
        "--workspace-root", 
        help="Root directory of the workspace"
    )
    parser.add_argument(
        "--exclude-dirs", 
        help="Comma-separated list of directories to exclude",
        default=".git,__pycache__,venv,env,.env,node_modules"
    )
    parser.add_argument(
        "--exclude-files", 
        help="Comma-separated list of files to exclude"
    )
    parser.add_argument(
        "--output", 
        help="Output file for the report (defaults to stdout)"
    )
    parser.add_argument(
        "--format", 
        choices=["json", "markdown"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    exclude_dirs = args.exclude_dirs.split(",") if args.exclude_dirs else None
    exclude_files = args.exclude_files.split(",") if args.exclude_files else None
    
    validator = DocReferenceValidator(
        index_file=args.index_file,
        workspace_root=args.workspace_root,
        exclude_dirs=exclude_dirs,
        exclude_files=exclude_files
    )
    
    report = validator.validate()
    
    if args.format == "json":
        output = report.to_json()
    else:
        output = report.to_markdown()
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        logger.info(f"Report written to {args.output}")
    else:
        print(output)
    
    # Return non-zero exit code if there are errors
    return 1 if report.error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main()) 