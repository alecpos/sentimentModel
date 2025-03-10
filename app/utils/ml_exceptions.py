"""
ML Exceptions Module

This module defines custom exceptions for ML-related error scenarios in the WITHIN ML Prediction System.
These exceptions provide standardized error handling across the API and ML components.
"""

from typing import Dict, List, Optional, Any


class MLBaseException(Exception):
    """Base exception for all ML-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, List[str]]] = None):
        """
        Initialize ML base exception.
        
        Args:
            message: Error message
            details: Optional dictionary with detailed error information
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ModelNotFoundError(MLBaseException):
    """Raised when a requested ML model is not found."""
    
    def __init__(self, message: str, details: Optional[Dict[str, List[str]]] = None):
        """
        Initialize model not found error.
        
        Args:
            message: Error message explaining which model was not found
            details: Optional dictionary with detailed error information
        """
        super().__init__(message, details)


class InvalidFeatureFormatError(MLBaseException):
    """Raised when features are not in the expected format."""
    
    def __init__(self, message: str, details: Optional[Dict[str, List[str]]] = None):
        """
        Initialize invalid feature format error.
        
        Args:
            message: Error message explaining the validation issue
            details: Dictionary with field-specific validation errors
        """
        super().__init__(message, details)


class PredictionFailedError(MLBaseException):
    """Raised when a prediction fails to complete."""
    
    def __init__(self, message: str, details: Optional[Dict[str, List[str]]] = None):
        """
        Initialize prediction failed error.
        
        Args:
            message: Error message explaining why prediction failed
            details: Optional dictionary with detailed error information
        """
        super().__init__(message, details)


class InsufficientDataError(MLBaseException):
    """Raised when there is not enough data for prediction."""
    
    def __init__(self, message: str, details: Optional[Dict[str, List[str]]] = None):
        """
        Initialize insufficient data error.
        
        Args:
            message: Error message explaining the data requirements
            details: Optional dictionary with data quality metrics
        """
        super().__init__(message, details)


class ModelTrainingError(MLBaseException):
    """Raised when there is an error during model training."""
    
    def __init__(self, message: str, details: Optional[Dict[str, List[str]]] = None):
        """
        Initialize model training error.
        
        Args:
            message: Error message explaining the training issue
            details: Optional dictionary with training error details
        """
        super().__init__(message, details)


class ModelVersionError(MLBaseException):
    """Raised when there is an issue with model versioning."""
    
    def __init__(self, message: str, details: Optional[Dict[str, List[str]]] = None):
        """
        Initialize model version error.
        
        Args:
            message: Error message explaining the version issue
            details: Optional dictionary with version details
        """
        super().__init__(message, details)


class ModelError(MLBaseException):
    """Raised when there is an internal model error."""
    
    def __init__(self, message: str, details: Optional[Dict[str, List[str]]] = None):
        """
        Initialize internal model error.
        
        Args:
            message: Error message explaining the internal error
            details: Optional dictionary with error details
        """
        super().__init__(message, details)


class FeatureEngineeringError(MLBaseException):
    """Raised when there is an error in feature engineering process."""
    
    def __init__(self, message: str, details: Optional[Dict[str, List[str]]] = None):
        """
        Initialize feature engineering error.
        
        Args:
            message: Error message explaining the feature engineering issue
            details: Optional dictionary with feature details
        """
        super().__init__(message, details)


class ExplanationError(MLBaseException):
    """Raised when there is an error generating model explanations."""
    
    def __init__(self, message: str, details: Optional[Dict[str, List[str]]] = None):
        """
        Initialize explanation error.
        
        Args:
            message: Error message explaining the explanation issue
            details: Optional dictionary with explanation details
        """
        super().__init__(message, details)