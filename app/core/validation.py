"""
Data Validation Module for WITHIN System

This module provides utilities for validating datasets against schemas to ensure
data quality and format requirements are met before model training and evaluation.
The module is designed to work with the WITHIN Ad Score & Account Health Predictor
system to ensure consistent data processing.

Implementation follows WITHIN ML Backend standards with strict type safety and
comprehensive documentation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable
import json
import logging
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class DataValidationSchema:
    """Schema for validating datasets.
    
    Attributes:
        name: Name of the schema
        version: Schema version
        description: Description of the schema
        required_columns: List of columns that must be present
        column_types: Dictionary mapping column names to expected types
        value_constraints: Dictionary of constraints for column values
        statistics: Dictionary of statistical constraints for columns
        fairness_constraints: Dictionary of fairness-related constraints
        preprocessing: Dictionary of preprocessing steps to apply
    """
    
    name: str
    version: str
    description: str
    required_columns: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    value_constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fairness_constraints: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, schema_path: Union[str, Path]) -> 'DataValidationSchema':
        """Load schema from a JSON file.
        
        Args:
            schema_path: Path to the schema file
            
        Returns:
            DataValidationSchema object
        """
        schema_path = Path(schema_path)
        
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
            
        try:
            with open(schema_path, "r") as f:
                schema_dict = json.load(f)
                
            # Required fields
            if "name" not in schema_dict:
                schema_dict["name"] = schema_path.stem
                
            if "version" not in schema_dict:
                schema_dict["version"] = "1.0.0"
                
            if "description" not in schema_dict:
                schema_dict["description"] = f"Schema loaded from {schema_path}"
                
            return cls(**schema_dict)
                
        except Exception as e:
            logger.error(f"Error loading schema from {schema_path}: {str(e)}")
            raise ValueError(f"Failed to load schema: {str(e)}")

def validate_dataset(df: pd.DataFrame, schema: DataValidationSchema) -> Dict[str, Any]:
    """Validate a dataframe against a schema.
    
    Args:
        df: Dataframe to validate
        schema: Validation schema
        
    Returns:
        Dictionary with validation results
    """
    errors = []
    warnings = []
    
    # Check required columns
    missing_columns = [col for col in schema.required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Check column types
    column_type_errors = []
    for col, expected_type in schema.column_types.items():
        if col not in df.columns:
            continue
            
        is_valid = _validate_column_type(df[col], expected_type)
        if not is_valid:
            actual_type = df[col].dtype
            column_type_errors.append(f"Column '{col}' has type '{actual_type}', expected '{expected_type}'")
    
    if column_type_errors:
        errors.extend(column_type_errors)
    
    # Check value constraints
    value_constraint_errors = []
    for col, constraints in schema.value_constraints.items():
        if col not in df.columns:
            continue
            
        col_errors = _validate_value_constraints(df[col], constraints)
        if col_errors:
            value_constraint_errors.extend([f"Column '{col}': {err}" for err in col_errors])
    
    if value_constraint_errors:
        errors.extend(value_constraint_errors)
    
    # Check statistical constraints
    stat_errors = []
    for col, constraints in schema.statistics.items():
        if col not in df.columns:
            continue
            
        col_errors = _validate_statistics(df[col], constraints)
        if col_errors:
            stat_errors.extend([f"Column '{col}': {err}" for err in col_errors])
    
    if stat_errors:
        errors.extend(stat_errors)
    
    # Apply schema preprocessing if needed
    if schema.preprocessing:
        df = apply_schema_preprocessing(df, schema.preprocessing)
    
    # Return validation results
    is_valid = len(errors) == 0
    
    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "schema_name": schema.name,
        "schema_version": schema.version
    }

def _validate_column_type(series: pd.Series, expected_type: str) -> bool:
    """Validate the type of a column.
    
    Args:
        series: Column to validate
        expected_type: Expected type
        
    Returns:
        True if valid, False otherwise
    """
    if expected_type == "int":
        return pd.api.types.is_integer_dtype(series) or series.apply(lambda x: isinstance(x, int)).all()
    elif expected_type == "float":
        return pd.api.types.is_float_dtype(series) or series.apply(lambda x: isinstance(x, (int, float))).all()
    elif expected_type == "str":
        return pd.api.types.is_string_dtype(series) or series.apply(lambda x: isinstance(x, str)).all()
    elif expected_type == "bool":
        return pd.api.types.is_bool_dtype(series) or series.apply(lambda x: isinstance(x, bool)).all()
    elif expected_type == "category":
        return pd.api.types.is_categorical_dtype(series)
    elif expected_type == "datetime":
        return pd.api.types.is_datetime64_dtype(series)
    else:
        return True  # Unknown type, assume valid

def _validate_value_constraints(series: pd.Series, constraints: Dict[str, Any]) -> List[str]:
    """Validate a column against value constraints.
    
    Args:
        series: Column to validate
        constraints: Dictionary of constraints
        
    Returns:
        List of error messages
    """
    errors = []
    
    # Check allowed values
    if "allowed_values" in constraints:
        allowed_values = constraints["allowed_values"]
        invalid_values = series[~series.isin(allowed_values) & ~series.isna()].unique()
        
        if len(invalid_values) > 0:
            errors.append(f"Invalid values: {list(invalid_values)}. Allowed values: {allowed_values}")
    
    # Check min/max constraints for numeric columns
    if pd.api.types.is_numeric_dtype(series):
        if "min" in constraints:
            min_val = constraints["min"]
            invalid_count = (series < min_val).sum()
            
            if invalid_count > 0:
                errors.append(f"{invalid_count} values below minimum {min_val}")
                
        if "max" in constraints:
            max_val = constraints["max"]
            invalid_count = (series > max_val).sum()
            
            if invalid_count > 0:
                errors.append(f"{invalid_count} values above maximum {max_val}")
    
    # Check string length for string columns
    if pd.api.types.is_string_dtype(series):
        if "min_length" in constraints:
            min_length = constraints["min_length"]
            invalid_count = (series.str.len() < min_length).sum()
            
            if invalid_count > 0:
                errors.append(f"{invalid_count} values shorter than {min_length} characters")
                
        if "max_length" in constraints:
            max_length = constraints["max_length"]
            invalid_count = (series.str.len() > max_length).sum()
            
            if invalid_count > 0:
                errors.append(f"{invalid_count} values longer than {max_length} characters")
    
    return errors

def _validate_statistics(series: pd.Series, constraints: Dict[str, Any]) -> List[str]:
    """Validate a column against statistical constraints.
    
    Args:
        series: Column to validate
        constraints: Dictionary of constraints
        
    Returns:
        List of error messages
    """
    errors = []
    
    # Skip empty or all-null columns
    if len(series) == 0 or series.isna().all():
        return errors
    
    # Check mean constraints
    if "mean" in constraints:
        expected_mean = constraints["mean"]
        tolerance = constraints.get("mean_tolerance", 0.1)
        
        actual_mean = series.mean()
        if abs(actual_mean - expected_mean) > tolerance:
            errors.append(f"Mean {actual_mean} differs from expected {expected_mean} by more than {tolerance}")
    
    # Check std constraints
    if "std" in constraints:
        expected_std = constraints["std"]
        tolerance = constraints.get("std_tolerance", 0.1)
        
        actual_std = series.std()
        if abs(actual_std - expected_std) > tolerance:
            errors.append(f"Std {actual_std} differs from expected {expected_std} by more than {tolerance}")
    
    # Check missing rate constraints
    if "max_missing_rate" in constraints:
        max_rate = constraints["max_missing_rate"]
        
        actual_rate = series.isna().mean()
        if actual_rate > max_rate:
            errors.append(f"Missing rate {actual_rate:.2f} exceeds maximum {max_rate}")
    
    return errors

def apply_schema_preprocessing(df: pd.DataFrame, preprocessing: Dict[str, Any]) -> pd.DataFrame:
    """Apply preprocessing steps defined in a schema.
    
    Args:
        df: Dataframe to preprocess
        preprocessing: Preprocessing configuration
        
    Returns:
        Preprocessed dataframe
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Apply column transformations
    for col, transforms in preprocessing.get("column_transforms", {}).items():
        if col not in result.columns:
            continue
            
        # Apply each transformation in sequence
        for transform in transforms:
            transform_type = transform.get("type")
            
            if transform_type == "fillna":
                value = transform.get("value")
                result[col] = result[col].fillna(value)
                
            elif transform_type == "replace":
                from_val = transform.get("from")
                to_val = transform.get("to")
                result[col] = result[col].replace(from_val, to_val)
                
            elif transform_type == "normalize":
                result[col] = (result[col] - result[col].mean()) / result[col].std()
                
            elif transform_type == "log":
                # Add small constant to handle zeros
                epsilon = transform.get("epsilon", 1e-10)
                result[col] = np.log(result[col] + epsilon)
    
    # Create derived columns
    for col, definition in preprocessing.get("derived_columns", {}).items():
        source_cols = definition.get("source_columns", [])
        
        # Skip if any source column is missing
        if not all(source_col in result.columns for source_col in source_cols):
            continue
            
        operation = definition.get("operation")
        
        if operation == "sum":
            result[col] = result[source_cols].sum(axis=1)
            
        elif operation == "mean":
            result[col] = result[source_cols].mean(axis=1)
            
        elif operation == "concat":
            separator = definition.get("separator", "_")
            result[col] = result[source_cols[0]].astype(str)
            
            for source_col in source_cols[1:]:
                result[col] = result[col] + separator + result[source_col].astype(str)
    
    return result 