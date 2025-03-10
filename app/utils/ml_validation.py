"""
ML Validation Module

This module provides Pydantic models and validators for ML-specific validation
in the WITHIN ML Prediction System.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel as PydanticBaseModel, Field, validator, root_validator
import numpy as np
import math


class BaseMLModel(PydanticBaseModel):
    """
    Base Pydantic model with enhanced validation capabilities for ML models.
    
    This model extends Pydantic's BaseModel to provide additional validation
    and configuration options specific to ML data.
    """
    
    class Config:
        """Configuration for ML models."""
        # Allow extra attributes during validation but exclude them from the model
        extra = "ignore"
        
        # Use more strict validation rules
        validate_assignment = True
        
        # Better error messages
        error_msg_templates = {
            "type_error": "Expected {expected_type} but received {input_type}",
            "value_error.missing": "Field is required",
        }
        
        # Use JSON schema with examples
        schema_extra = {
            "examples": []
        }


class PredictionRequest(BaseMLModel):
    """
    Base model for prediction requests.
    
    This model provides common fields and validation for all prediction requests.
    """
    
    model_id: str = Field(
        ...,
        description="The ID of the model to use for prediction",
        example="ad-score-predictor"
    )
    include_explanation: bool = Field(
        False,
        description="Whether to include feature importance explanations in the response",
        example=True
    )


def validate_numeric_features(
    numeric_features: Optional[List[float]],
    min_features: int = 3,
    min_value: float = 0.0,
    max_value: float = 1.0
) -> Optional[List[float]]:
    """
    Validate numeric features according to constraints.
    
    Args:
        numeric_features: List of numeric features to validate
        min_features: Minimum number of features required
        min_value: Minimum value allowed for features
        max_value: Maximum value allowed for features
        
    Returns:
        Validated numeric features
        
    Raises:
        ValueError: If validation fails
    """
    # Check if features are provided
    if numeric_features is None:
        return None
    
    # Check number of features
    if len(numeric_features) < min_features:
        raise ValueError(f"Must contain at least {min_features} features")
    
    # Check for NaN or infinity
    if any(math.isnan(x) or math.isinf(x) for x in numeric_features):
        raise ValueError("Numeric features cannot contain NaN or infinity")
    
    # Check value range
    if any(not (min_value <= x <= max_value) for x in numeric_features):
        raise ValueError(f"Numeric features must be between {min_value} and {max_value}")
    
    return numeric_features


def validate_text_features(
    text_features: Optional[List[str]],
    max_length: int = 1000
) -> Optional[List[str]]:
    """
    Validate text features according to constraints.
    
    Args:
        text_features: List of text features to validate
        max_length: Maximum length allowed for each text feature
        
    Returns:
        Validated text features
        
    Raises:
        ValueError: If validation fails
    """
    # Check if features are provided
    if text_features is None:
        return None
    
    # Check for empty values
    if any(not text.strip() for text in text_features):
        raise ValueError("Text features cannot be empty")
    
    # Check length constraints
    if any(len(text) > max_length for text in text_features):
        raise ValueError(f"Text features cannot exceed {max_length} characters")
    
    return text_features


def validate_categorical_features(
    categorical_features: Optional[List[str]],
    allowed_categories: Optional[List[str]] = None
) -> Optional[List[str]]:
    """
    Validate categorical features according to constraints.
    
    Args:
        categorical_features: List of categorical features to validate
        allowed_categories: Optional list of allowed category values
        
    Returns:
        Validated categorical features
        
    Raises:
        ValueError: If validation fails
    """
    # Check if features are provided
    if categorical_features is None:
        return None
    
    # Check for empty values
    if any(not category.strip() for category in categorical_features):
        raise ValueError("Categorical features cannot be empty")
    
    # Check allowed categories if specified
    if allowed_categories is not None:
        invalid_categories = [c for c in categorical_features if c not in allowed_categories]
        if invalid_categories:
            raise ValueError(
                f"Invalid categories: {', '.join(invalid_categories)}. "
                f"Allowed categories: {', '.join(allowed_categories)}"
            )
    
    return categorical_features


def validate_has_features(values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that at least one feature type is provided.
    
    Args:
        values: Dictionary of values to validate
        
    Returns:
        Validated values
        
    Raises:
        ValueError: If no feature type is provided
    """
    has_features = any([
        values.get('text_features'),
        values.get('numeric_features'),
        values.get('categorical_features'),
        values.get('image_features')
    ])
    
    if not has_features:
        raise ValueError("At least one feature type must be provided")
        
    return values


class EnhancedAdScorePredictionRequest(PredictionRequest):
    """
    Enhanced request model for ad score prediction with comprehensive validation.
    """
    
    model_id: str = "ad-score-predictor"
    text_features: Optional[List[str]] = None
    numeric_features: Optional[List[float]] = None
    categorical_features: Optional[List[str]] = None
    image_features: Optional[List[str]] = None
    include_explanation: bool = False
    
    # Numeric feature validation
    @validator('numeric_features')
    def validate_numeric_features(cls, v):
        """Validate numeric features with detailed constraints."""
        return validate_numeric_features(v, min_features=3, min_value=0.0, max_value=1.0)
    
    # Text feature validation
    @validator('text_features')
    def validate_text_features(cls, v):
        """Validate text features with detailed constraints."""
        return validate_text_features(v, max_length=1000)
    
    # Categorical feature validation
    @validator('categorical_features')
    def validate_categorical_features(cls, v):
        """Validate categorical features with detailed constraints."""
        return validate_categorical_features(v)
    
    # Cross-feature validation
    @root_validator
    def validate_feature_combinations(cls, values):
        """Validate feature combinations with business rules."""
        # Ensure at least one feature type is provided
        validate_has_features(values)
        
        # Check specific combinations based on business rules
        has_text = bool(values.get('text_features'))
        has_image = bool(values.get('image_features'))
        
        if has_image and not has_text:
            raise ValueError("Text features must be provided when image features are present")
            
        return values 