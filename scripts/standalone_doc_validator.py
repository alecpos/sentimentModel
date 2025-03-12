#!/usr/bin/env python
"""
Standalone Documentation Structure Validator

This script checks the documentation structure across the WITHIN ML Prediction System
codebase without requiring imports from the application code. It verifies that:

1. Each directory has a README.md file
2. Each Python package has an __init__.py file
3. Documentation in README.md and __init__.py is consistent

Usage:
    python scripts/standalone_doc_validator.py [options]

Options:
    --root-dir PATH       Root directory to start validation (default: app/)
    --output PATH         File to write the report to (default: stdout)
    --format FORMAT       Output format: text, json, markdown (default: text)
    --verbose             Enable detailed logging
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("doc_structure_validator")


class IssueType(Enum):
    """Types of documentation issues."""
    MISSING_README = "missing_readme"
    MISSING_INIT = "missing_init"
    EMPTY_README = "empty_readme"
    EMPTY_INIT = "empty_init"
    MISSING_DOCSTRING = "missing_docstring"
    INCOMPLETE_DOCSTRING = "incomplete_docstring"
    INCONSISTENT_EXPORTS = "inconsistent_exports"
    PLANNED_NOT_IMPLEMENTED = "planned_not_implemented"
    IMPLEMENTED_NOT_DOCUMENTED = "implemented_not_documented"


class IssueSeverity(Enum):
    """Severity levels for documentation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class DocumentationIssue:
    """Represents an issue found during documentation validation."""
    path: Path
    issue_type: IssueType
    description: str
    severity: IssueSeverity
    suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    """Report generated after validation."""
    issues: List[DocumentationIssue] = field(default_factory=list)
    directories_checked: int = 0
    readme_files_found: int = 0
    init_files_found: int = 0
    
    @property
    def error_count(self) -> int:
        """Count of error issues."""
        return sum(1 for issue in self.issues if issue.severity == IssueSeverity.ERROR)
    
    @property
    def warning_count(self) -> int:
        """Count of warning issues."""
        return sum(1 for issue in self.issues if issue.severity == IssueSeverity.WARNING)
    
    @property
    def info_count(self) -> int:
        """Count of info issues."""
        return sum(1 for issue in self.issues if issue.severity == IssueSeverity.INFO)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "summary": {
                "directories_checked": self.directories_checked,
                "readme_files_found": self.readme_files_found,
                "init_files_found": self.init_files_found,
                "total_issues": len(self.issues),
                "error_count": self.error_count,
                "warning_count": self.warning_count,
                "info_count": self.info_count
            },
            "issues": [
                {
                    "path": str(issue.path),
                    "issue_type": issue.issue_type.value,
                    "description": issue.description,
                    "severity": issue.severity.value,
                    "suggestion": issue.suggestion
                }
                for issue in self.issues
            ]
        }
    
    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_text(self) -> str:
        """Convert report to text format."""
        text = []
        text.append(f"Documentation Structure Validation Report")
        text.append(f"=======================================")
        text.append(f"")
        text.append(f"Summary:")
        text.append(f"- Directories checked: {self.directories_checked}")
        text.append(f"- README.md files found: {self.readme_files_found}")
        text.append(f"- __init__.py files found: {self.init_files_found}")
        text.append(f"- Total issues: {len(self.issues)}")
        text.append(f"- Errors: {self.error_count}")
        text.append(f"- Warnings: {self.warning_count}")
        text.append(f"- Info: {self.info_count}")
        text.append(f"")
        
        if self.issues:
            text.append(f"Issues:")
            for i, issue in enumerate(self.issues, 1):
                text.append(f"{i}. {issue.severity.value.upper()}: {issue.path}")
                text.append(f"   Type: {issue.issue_type.value}")
                text.append(f"   Description: {issue.description}")
                if issue.suggestion:
                    text.append(f"   Suggestion: {issue.suggestion}")
                text.append(f"")
        else:
            text.append(f"No issues found!")
        
        return "\n".join(text)
    
    def to_markdown(self) -> str:
        """Convert report to Markdown format."""
        md = []
        md.append(f"# Documentation Structure Validation Report")
        md.append(f"")
        md.append(f"## Summary")
        md.append(f"")
        md.append(f"- **Directories checked:** {self.directories_checked}")
        md.append(f"- **README.md files found:** {self.readme_files_found}")
        md.append(f"- **__init__.py files found:** {self.init_files_found}")
        md.append(f"- **Total issues:** {len(self.issues)}")
        md.append(f"- **Errors:** {self.error_count}")
        md.append(f"- **Warnings:** {self.warning_count}")
        md.append(f"- **Info:** {self.info_count}")
        md.append(f"")
        
        if self.issues:
            md.append(f"## Issues")
            md.append(f"")
            
            # Group issues by severity
            errors = [issue for issue in self.issues if issue.severity == IssueSeverity.ERROR]
            warnings = [issue for issue in self.issues if issue.severity == IssueSeverity.WARNING]
            infos = [issue for issue in self.issues if issue.severity == IssueSeverity.INFO]
            
            if errors:
                md.append(f"### Errors")
                md.append(f"")
                md.append(f"| Path | Issue Type | Description | Suggestion |")
                md.append(f"|------|------------|-------------|------------|")
                for issue in errors:
                    suggestion = issue.suggestion or "N/A"
                    md.append(f"| `{issue.path}` | {issue.issue_type.value} | {issue.description} | {suggestion} |")
                md.append(f"")
            
            if warnings:
                md.append(f"### Warnings")
                md.append(f"")
                md.append(f"| Path | Issue Type | Description | Suggestion |")
                md.append(f"|------|------------|-------------|------------|")
                for issue in warnings:
                    suggestion = issue.suggestion or "N/A"
                    md.append(f"| `{issue.path}` | {issue.issue_type.value} | {issue.description} | {suggestion} |")
                md.append(f"")
            
            if infos:
                md.append(f"### Information")
                md.append(f"")
                md.append(f"| Path | Issue Type | Description | Suggestion |")
                md.append(f"|------|------------|-------------|------------|")
                for issue in infos:
                    suggestion = issue.suggestion or "N/A"
                    md.append(f"| `{issue.path}` | {issue.issue_type.value} | {issue.description} | {suggestion} |")
                md.append(f"")
        else:
            md.append(f"## No issues found!")
            md.append(f"")
            md.append(f"All directories have proper documentation structure.")
        
        return "\n".join(md)


class DocumentationValidator:
    """Validator for documentation structure."""
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        exclude_dirs: Optional[List[str]] = None,
        exclude_files: Optional[List[str]] = None
    ):
        """
        Initialize the validator.
        
        Args:
            root_dir: Root directory to validate
            exclude_dirs: Directories to exclude
            exclude_files: Files to exclude
        """
        self.root_dir = Path(root_dir).resolve()
        self.exclude_dirs = exclude_dirs or [".git", "__pycache__", "venv", ".venv"]
        self.exclude_files = exclude_files or [".DS_Store"]
        self.report = ValidationReport()
    
    def validate(self) -> ValidationReport:
        """
        Validate the documentation structure.
        
        Returns:
            ValidationReport: The validation report
        """
        logger.info(f"Starting documentation structure validation in {self.root_dir}")
        
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # Skip excluded directories
            dirnames[:] = [d for d in dirnames if d not in self.exclude_dirs and not d.startswith(".")]
            
            current_dir = Path(dirpath)
            self.report.directories_checked += 1
            
            is_package = "__init__.py" in filenames
            has_readme = "README.md" in filenames
            
            # Check for README.md
            if has_readme:
                self.report.readme_files_found += 1
                readme_path = current_dir / "README.md"
                self._validate_readme(readme_path)
            else:
                # Only report missing README for non-empty directories or packages
                if is_package or len(filenames) > 0:
                    self.report.issues.append(
                        DocumentationIssue(
                            path=current_dir,
                            issue_type=IssueType.MISSING_README,
                            description=f"Directory does not have a README.md file",
                            severity=IssueSeverity.WARNING,
                            suggestion=f"Create a README.md file describing the purpose of this directory"
                        )
                    )
            
            # Check for __init__.py if it's a package
            python_files = [f for f in filenames if f.endswith(".py") and f != "__init__.py"]
            if python_files and not is_package:
                # This looks like a Python module, but no __init__.py
                self.report.issues.append(
                    DocumentationIssue(
                        path=current_dir,
                        issue_type=IssueType.MISSING_INIT,
                        description=f"Directory contains Python files but no __init__.py",
                        severity=IssueSeverity.ERROR,
                        suggestion=f"Create an __init__.py file to make this a proper Python package"
                    )
                )
            elif is_package:
                self.report.init_files_found += 1
                init_path = current_dir / "__init__.py"
                self._validate_init(init_path)
                
                # Check consistency between README and __init__
                if has_readme:
                    self._check_consistency(current_dir / "README.md", init_path)
        
        logger.info(f"Documentation structure validation complete")
        logger.info(f"Found {len(self.report.issues)} issues ({self.report.error_count} errors, {self.report.warning_count} warnings)")
        
        return self.report
    
    def _validate_readme(self, readme_path: Path) -> None:
        """Validate a README.md file."""
        try:
            content = readme_path.read_text()
            if not content.strip():
                self.report.issues.append(
                    DocumentationIssue(
                        path=readme_path,
                        issue_type=IssueType.EMPTY_README,
                        description=f"README.md file is empty",
                        severity=IssueSeverity.ERROR,
                        suggestion=f"Add content describing the purpose of this directory"
                    )
                )
            elif len(content.strip().split("\n")) < 3:
                self.report.issues.append(
                    DocumentationIssue(
                        path=readme_path,
                        issue_type=IssueType.INCOMPLETE_DOCSTRING,
                        description=f"README.md file is too short (less than 3 lines)",
                        severity=IssueSeverity.WARNING,
                        suggestion=f"Expand the README.md with more information"
                    )
                )
        except Exception as e:
            logger.error(f"Error reading {readme_path}: {e}")
            self.report.issues.append(
                DocumentationIssue(
                    path=readme_path,
                    issue_type=IssueType.EMPTY_README,
                    description=f"Error reading README.md: {e}",
                    severity=IssueSeverity.ERROR,
                    suggestion=f"Fix the README.md file"
                )
            )
    
    def _validate_init(self, init_path: Path) -> None:
        """Validate an __init__.py file."""
        try:
            content = init_path.read_text()
            if not content.strip():
                self.report.issues.append(
                    DocumentationIssue(
                        path=init_path,
                        issue_type=IssueType.EMPTY_INIT,
                        description=f"__init__.py file is empty",
                        severity=IssueSeverity.WARNING,
                        suggestion=f"Add a docstring and necessary imports"
                    )
                )
            else:
                # Check for module docstring
                docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
                if not docstring_match:
                    self.report.issues.append(
                        DocumentationIssue(
                            path=init_path,
                            issue_type=IssueType.MISSING_DOCSTRING,
                            description=f"__init__.py file lacks a module docstring",
                            severity=IssueSeverity.WARNING,
                            suggestion=f"Add a docstring describing the package"
                        )
                    )
                elif len(docstring_match.group(1).strip().split("\n")) < 2:
                    self.report.issues.append(
                        DocumentationIssue(
                            path=init_path,
                            issue_type=IssueType.INCOMPLETE_DOCSTRING,
                            description=f"Module docstring is too short (less than 2 lines)",
                            severity=IssueSeverity.INFO,
                            suggestion=f"Expand the docstring with more information"
                        )
                    )
                
                # Check for placeholder text
                placeholder_patterns = [
                    r"placeholder",
                    r"to be implemented",
                    r"will be populated",
                    r"planned"
                ]
                if any(re.search(pattern, content, re.IGNORECASE) for pattern in placeholder_patterns):
                    self.report.issues.append(
                        DocumentationIssue(
                            path=init_path,
                            issue_type=IssueType.PLANNED_NOT_IMPLEMENTED,
                            description=f"__init__.py contains placeholder text for planned components",
                            severity=IssueSeverity.INFO,
                            suggestion=f"Implement the planned components or update the docstring"
                        )
                    )
        except Exception as e:
            logger.error(f"Error reading {init_path}: {e}")
            self.report.issues.append(
                DocumentationIssue(
                    path=init_path,
                    issue_type=IssueType.EMPTY_INIT,
                    description=f"Error reading __init__.py: {e}",
                    severity=IssueSeverity.ERROR,
                    suggestion=f"Fix the __init__.py file"
                )
            )
    
    def _check_consistency(self, readme_path: Path, init_path: Path) -> None:
        """Check consistency between README.md and __init__.py."""
        try:
            readme_content = readme_path.read_text()
            init_content = init_path.read_text()
            
            # Extract module name from the directory path
            module_name = init_path.parent.name
            
            # Look for components mentioned in README but not in __init__
            # This is a simple check for pattern matching; a more sophisticated
            # approach would use AST parsing to identify actual exports
            
            # Extract class and function names from README
            readme_components = set()
            class_matches = re.finditer(r'class\s+([A-Za-z0-9_]+)', readme_content)
            for match in class_matches:
                readme_components.add(match.group(1))
            
            function_matches = re.finditer(r'def\s+([A-Za-z0-9_]+)', readme_content)
            for match in function_matches:
                readme_components.add(match.group(1))
            
            # Extract components from code blocks in README
            code_block_pattern = r'```.*?```'
            code_blocks = re.finditer(code_block_pattern, readme_content, re.DOTALL)
            for block in code_blocks:
                block_text = block.group(0)
                # Look for patterns like ClassName or function_name
                component_matches = re.finditer(r'([A-Za-z0-9_]+)\(', block_text)
                for match in component_matches:
                    component = match.group(1)
                    # Filter out common Python functions
                    if component not in ['print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set']:
                        readme_components.add(component)
            
            # Extract exports from __init__.py
            exports = set()
            all_match = re.search(r'__all__\s*=\s*\[(.*?)\]', init_content, re.DOTALL)
            if all_match:
                all_content = all_match.group(1)
                quote_patterns = ['"([^"]+)"', "'([^']+)'"]
                for pattern in quote_patterns:
                    for match in re.finditer(pattern, all_content):
                        exports.add(match.group(1))
            
            # Check for imports
            import_pattern = r'from\s+\.\s*(\w+)\s+import\s+(.*)'
            for match in re.finditer(import_pattern, init_content):
                imported_module = match.group(1)
                imported_items = match.group(2)
                
                # Extract individual items
                for item in re.split(r',\s*', imported_items):
                    item = item.strip()
                    if item and item != '*':
                        exports.add(item)
            
            # Check for components mentioned in README but not exported
            mentioned_but_not_exported = readme_components - exports
            if mentioned_but_not_exported and exports:  # Only warn if there are exports
                self.report.issues.append(
                    DocumentationIssue(
                        path=init_path,
                        issue_type=IssueType.INCONSISTENT_EXPORTS,
                        description=f"Components mentioned in README.md but not exported in __init__.py: {', '.join(mentioned_but_not_exported)}",
                        severity=IssueSeverity.WARNING,
                        suggestion=f"Update __init__.py to export these components or update README.md"
                    )
                )
        except Exception as e:
            logger.error(f"Error checking consistency between {readme_path} and {init_path}: {e}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate documentation structure")
    parser.add_argument(
        "--root-dir",
        default="app",
        help="Root directory to start validation from"
    )
    parser.add_argument(
        "--exclude-dirs",
        help="Comma-separated list of directories to exclude",
        default=".git,__pycache__,venv,.venv,build,dist"
    )
    parser.add_argument(
        "--exclude-files",
        help="Comma-separated list of files to exclude",
        default=".DS_Store"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format"
    )
    parser.add_argument(
        "--output",
        help="File to write the report to (default: stdout)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Parse exclude lists
    exclude_dirs = [d.strip() for d in args.exclude_dirs.split(",") if d.strip()]
    exclude_files = [f.strip() for f in args.exclude_files.split(",") if f.strip()]
    
    # Initialize validator
    validator = DocumentationValidator(
        root_dir=args.root_dir,
        exclude_dirs=exclude_dirs,
        exclude_files=exclude_files
    )
    
    # Run validation
    report = validator.validate()
    
    # Generate output
    if args.format == "json":
        output = report.to_json()
    elif args.format == "markdown":
        output = report.to_markdown()
    else:  # text
        output = report.to_text()
    
    # Write output
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report written to {args.output}")
    else:
        print(output)
    
    # Return exit code
    return 1 if report.error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main()) 