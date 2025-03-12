"""
Schema validation utilities for the WITHIN ML Prediction System.

This module provides a schema validation framework to ensure data consistency
and integrity before processing in ML pipelines. It supports JSON schema-based
validation with additional capabilities for ML-specific constraints.
"""

import json
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set
from pathlib import Path

logger = logging.getLogger(__name__)

class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationIssue:
    """Represents a single validation issue found during schema validation."""
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    column: Optional[str] = None
    row_index: Optional[int] = None
    constraint_type: Optional[str] = None
    actual_value: Optional[Any] = None
    expected_value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation issue to a dictionary."""
        return {
            "message": self.message,
            "severity": self.severity.value,
            "column": self.column,
            "row_index": self.row_index,
            "constraint_type": self.constraint_type,
            "actual_value": str(self.actual_value) if self.actual_value is not None else None,
            "expected_value": str(self.expected_value) if self.expected_value is not None else None
        }

@dataclass
class ValidationResult:
    """Result of a schema validation operation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation result to a dictionary."""
        return {
            "is_valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "stats": self.stats
        }

    def __str__(self) -> str:
        """String representation of the validation result."""
        if self.is_valid:
            return f"Validation passed: {len(self.issues)} warnings/info"
        return f"Validation failed: {len(self.issues)} issues found"

@dataclass
class DataValidationSchema:
    """Schema definition for validating datasets in ML pipelines."""
    name: str
    version: str
    description: str
    required_columns: List[str]
    column_types: Dict[str, str]
    value_constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fairness_constraints: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'DataValidationSchema':
        """Load schema from a JSON file.
        
        Args:
            json_path: Path to the JSON schema file
            
        Returns:
            DataValidationSchema: The loaded schema
        """
        with open(json_path, 'r') as f:
            schema_dict = json.load(f)
        
        return cls(
            name=schema_dict.get('name', ''),
            version=schema_dict.get('version', '1.0.0'),
            description=schema_dict.get('description', ''),
            required_columns=schema_dict.get('required_columns', []),
            column_types=schema_dict.get('column_types', {}),
            value_constraints=schema_dict.get('value_constraints', {}),
            statistics=schema_dict.get('statistics', {}),
            fairness_constraints=schema_dict.get('fairness_constraints', {})
        )
    
    def to_json(self, json_path: Union[str, Path]) -> None:
        """Save schema to a JSON file.
        
        Args:
            json_path: Path to save the JSON schema
        """
        schema_dict = {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'required_columns': self.required_columns,
            'column_types': self.column_types,
            'value_constraints': self.value_constraints,
            'statistics': self.statistics,
            'fairness_constraints': self.fairness_constraints
        }
        
        with open(json_path, 'w') as f:
            json.dump(schema_dict, f, indent=2)
    
    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """Validate a pandas DataFrame against this schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult: Result of the validation
        """
        issues = []
        stats = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "required_columns_present": 0,
            "column_type_matches": 0
        }
        
        # Check required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            for col in missing_columns:
                issues.append(ValidationIssue(
                    message=f"Required column '{col}' is missing",
                    severity=ValidationSeverity.ERROR,
                    column=col,
                    constraint_type="required_column"
                ))
        stats["required_columns_present"] = len(self.required_columns) - len(missing_columns)
        
        # Check column types
        for col, expected_type in self.column_types.items():
            if col not in df.columns:
                continue
                
            actual_type = self._get_column_type(df[col])
            type_match = self._check_type_compatibility(actual_type, expected_type)
            
            if not type_match:
                issues.append(ValidationIssue(
                    message=f"Column '{col}' has type '{actual_type}' but expected '{expected_type}'",
                    severity=ValidationSeverity.ERROR,
                    column=col,
                    constraint_type="column_type",
                    actual_value=actual_type,
                    expected_value=expected_type
                ))
            else:
                stats["column_type_matches"] += 1
        
        # Check value constraints
        for col, constraints in self.value_constraints.items():
            if col not in df.columns:
                continue
                
            # Skip empty/NA values for constraints
            valid_values = df[col].dropna()
            
            # Check allowed values
            if "allowed_values" in constraints:
                allowed_values = set(constraints["allowed_values"])
                unique_values = set(valid_values.unique())
                invalid_values = unique_values - allowed_values
                
                if invalid_values:
                    issues.append(ValidationIssue(
                        message=f"Column '{col}' contains {len(invalid_values)} invalid values not in allowed set",
                        severity=ValidationSeverity.ERROR,
                        column=col,
                        constraint_type="allowed_values",
                        actual_value=list(invalid_values)[:5],  # First 5 invalid values
                        expected_value=list(allowed_values)
                    ))
            
            # Check min/max for numeric columns
            if "min" in constraints and valid_values.size > 0:
                min_val = constraints["min"]
                if valid_values.min() < min_val:
                    issues.append(ValidationIssue(
                        message=f"Column '{col}' contains values below minimum {min_val}",
                        severity=ValidationSeverity.ERROR,
                        column=col,
                        constraint_type="min_value",
                        actual_value=valid_values.min(),
                        expected_value=min_val
                    ))
                    
            if "max" in constraints and valid_values.size > 0:
                max_val = constraints["max"]
                if valid_values.max() > max_val:
                    issues.append(ValidationIssue(
                        message=f"Column '{col}' contains values above maximum {max_val}",
                        severity=ValidationSeverity.ERROR,
                        column=col,
                        constraint_type="max_value",
                        actual_value=valid_values.max(),
                        expected_value=max_val
                    ))
        
        # Check statistics
        for col, stat_constraints in self.statistics.items():
            if col not in df.columns:
                continue
                
            column_data = df[col]
            
            if "min_length" in stat_constraints and column_data.dtype == object:
                min_length = stat_constraints["min_length"]
                string_lengths = column_data.dropna().astype(str).str.len()
                if string_lengths.min() < min_length:
                    issues.append(ValidationIssue(
                        message=f"Column '{col}' has strings shorter than minimum length {min_length}",
                        severity=ValidationSeverity.ERROR,
                        column=col,
                        constraint_type="min_length",
                        actual_value=string_lengths.min(),
                        expected_value=min_length
                    ))
                    
            if "max_length" in stat_constraints and column_data.dtype == object:
                max_length = stat_constraints["max_length"]
                string_lengths = column_data.dropna().astype(str).str.len()
                if string_lengths.max() > max_length:
                    issues.append(ValidationIssue(
                        message=f"Column '{col}' has strings longer than maximum length {max_length}",
                        severity=ValidationSeverity.ERROR,
                        column=col,
                        constraint_type="max_length",
                        actual_value=string_lengths.max(),
                        expected_value=max_length
                    ))
                    
            if "max_missing_pct" in stat_constraints:
                max_missing = stat_constraints["max_missing_pct"]
                missing_pct = (column_data.isna().sum() / len(column_data)) * 100
                if missing_pct > max_missing:
                    issues.append(ValidationIssue(
                        message=f"Column '{col}' has {missing_pct:.2f}% missing values, exceeding {max_missing}%",
                        severity=ValidationSeverity.ERROR,
                        column=col,
                        constraint_type="max_missing_pct",
                        actual_value=missing_pct,
                        expected_value=max_missing
                    ))
        
        # Determine if validation passed
        is_valid = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            stats=stats
        )
    
    def _get_column_type(self, column: pd.Series) -> str:
        """Get the simplified type name for a pandas Series."""
        dtype = column.dtype
        
        if pd.api.types.is_integer_dtype(dtype):
            return "int"
        elif pd.api.types.is_float_dtype(dtype):
            return "float"
        elif pd.api.types.is_bool_dtype(dtype):
            return "bool"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"
        elif pd.api.types.is_categorical_dtype(dtype):
            return "category"
        else:
            return "str"
    
    def _check_type_compatibility(self, actual_type: str, expected_type: str) -> bool:
        """Check if the actual type is compatible with the expected type."""
        # Direct match
        if actual_type == expected_type:
            return True
        
        # Numeric type compatibility
        if expected_type == "numeric" and actual_type in ("int", "float"):
            return True
        
        # String type compatibility 
        if expected_type == "str" and actual_type == "category":
            return True
        
        return False

def validate_dataset(df: pd.DataFrame, schema: DataValidationSchema) -> ValidationResult:
    """Validate a dataset against the provided schema.
    
    Args:
        df: DataFrame to validate
        schema: Schema to validate against
        
    Returns:
        ValidationResult: Result of the validation
    """
    return schema.validate_dataframe(df) 