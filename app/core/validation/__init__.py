"""
Data validation components for the WITHIN ML Prediction System.

This module provides validation utilities for ensuring data quality and integrity
throughout the ML pipeline. It includes components for schema validation, business
rule enforcement, and data quality checks before data enters the ML processing flow.

Key functionality includes:
- Schema validation for input data structures
- Business rule validation for domain-specific constraints
- Data quality metrics calculation and threshold validation
- Type checking and conversion
- Cross-field validation and conditional rules
"""

__version__ = "0.1.0"  # Pre-release version

# Package-level constants
VALIDATION_MODES = ["strict", "warn", "lenient"]
DEFAULT_VALIDATION_MODE = "strict"
DEFAULT_ERROR_LIMIT = 100  # Maximum number of validation errors to report

# Import and export validation components
from app.core.validation.schema_validator import (
    DataValidationSchema, validate_dataset, ValidationResult, ValidationIssue
)

# Export data quality validator
from app.core.validation.data_quality_validator import DataQualityValidator

__all__ = [
    "DataValidationSchema",
    "validate_dataset",
    "ValidationResult",
    "ValidationIssue",
    "DataQualityValidator",
    "VALIDATION_MODES",
    "DEFAULT_VALIDATION_MODE",
    "DEFAULT_ERROR_LIMIT",
] 