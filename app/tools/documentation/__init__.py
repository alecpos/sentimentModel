"""
Documentation tools for the WITHIN ML Prediction System.

This module provides utilities for validating, generating, and managing
documentation for the ML system. It includes tools for checking documentation
references, ensuring documentation consistency, and identifying missing documentation.

Key functionality includes:
- Documentation reference validation
- Documentation structure checking
- Documentation generation from code
- Documentation consistency validation
- Missing documentation identification

DOCUMENTATION STATUS: COMPLETE
"""

__version__ = "0.1.0"  # Pre-release version

# Export documentation validator
from .doc_reference_validator import (
    DocReferenceValidator,
    ValidationSeverity,
    ValidationReport,
    ValidationIssue
)

# Export documentation structure checker
from .documentation_checker import DocumentationChecker

__all__ = [
    "DocReferenceValidator",
    "ValidationSeverity",
    "ValidationReport",
    "ValidationIssue",
    "DocumentationChecker"
] 