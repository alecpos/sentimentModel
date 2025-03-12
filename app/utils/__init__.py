"""
Utility functions and helpers for the WITHIN ML Prediction System.

This package contains utility functions used across the application:
- Explainability tools for ML models
- Validation utilities for ML models
- API response formatting utilities
- Custom exception types for ML operations
"""

from .explainability import (
    generate_feature_importance,
    generate_shap_values,
    create_explanation,
    ExplanationFormat
)

from .ml_validation import (
    validate_model_inputs,
    validate_prediction_outputs,
    validate_model_performance
)

from .api_responses import (
    create_success_response,
    create_error_response,
    format_prediction_response,
    PaginatedResponse
)

from .ml_exceptions import (
    MLException,
    ModelNotFoundException,
    InvalidInputException,
    PredictionException,
    ModelTrainingException
)

__all__ = [
    # Explainability
    "generate_feature_importance",
    "generate_shap_values",
    "create_explanation",
    "ExplanationFormat",
    
    # ML Validation
    "validate_model_inputs",
    "validate_prediction_outputs",
    "validate_model_performance",
    
    # API Responses
    "create_success_response",
    "create_error_response",
    "format_prediction_response",
    "PaginatedResponse",
    
    # ML Exceptions
    "MLException",
    "ModelNotFoundException",
    "InvalidInputException",
    "PredictionException",
    "ModelTrainingException"
]
