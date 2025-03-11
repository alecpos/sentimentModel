"""
Custom exception classes for the ML system.

This module defines domain-specific exceptions for handling errors
in the ML prediction system, improving error handling and debugging.
"""
from typing import Dict, Any, List, Optional, Union
from datetime import datetime


class BaseMLError(Exception):
    """Base class for all ML system errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the base ML error.
        
        Args:
            message: Error message
            details: Optional dictionary of error details
        """
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class ModelNotFoundError(BaseMLError):
    """Error raised when a requested model cannot be found."""
    
    def __init__(self, model_id: str, message: Optional[str] = None):
        """
        Initialize the model not found error.
        
        Args:
            model_id: ID of the model that was not found
            message: Optional error message
        """
        message = message or f"Model with ID '{model_id}' not found"
        super().__init__(message, {"model_id": model_id})


class ModelLoadError(BaseMLError):
    """Error raised when a model cannot be loaded."""
    
    def __init__(self, model_id: str, reason: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the model load error.
        
        Args:
            model_id: ID of the model that could not be loaded
            reason: Reason for the loading failure
            details: Optional dictionary of error details
        """
        message = f"Failed to load model '{model_id}': {reason}"
        error_details = details or {}
        error_details["model_id"] = model_id
        error_details["reason"] = reason
        super().__init__(message, error_details)


class PredictionError(BaseMLError):
    """Error raised when a prediction fails."""
    
    def __init__(self, model_id: str, reason: str, input_data: Optional[Any] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the prediction error.
        
        Args:
            model_id: ID of the model that failed to predict
            reason: Reason for the prediction failure
            input_data: Optional input data that caused the failure
            details: Optional dictionary of error details
        """
        message = f"Prediction failed for model '{model_id}': {reason}"
        error_details = details or {}
        error_details["model_id"] = model_id
        error_details["reason"] = reason
        
        if input_data is not None:
            try:
                # Attempt to serialize input data for debugging
                if hasattr(input_data, "shape"):
                    error_details["input_shape"] = list(input_data.shape)
                elif hasattr(input_data, "__len__"):
                    error_details["input_length"] = len(input_data)
            except:
                pass
                
        super().__init__(message, error_details)


class ValidationError(BaseMLError):
    """Error raised when validation fails."""
    
    def __init__(self, reason: str, validation_results: Optional[Dict[str, Any]] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the validation error.
        
        Args:
            reason: Reason for the validation failure
            validation_results: Optional dictionary of validation results
            details: Optional dictionary of error details
        """
        message = f"Validation failed: {reason}"
        error_details = details or {}
        
        if validation_results:
            error_details["validation_results"] = validation_results
            
        super().__init__(message, error_details)


class FairnessError(BaseMLError):
    """Error raised when fairness evaluation fails."""
    
    def __init__(self, reason: str, fairness_metrics: Optional[Dict[str, Any]] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the fairness error.
        
        Args:
            reason: Reason for the fairness failure
            fairness_metrics: Optional dictionary of fairness metrics
            details: Optional dictionary of error details
        """
        message = f"Fairness evaluation failed: {reason}"
        error_details = details or {}
        
        if fairness_metrics:
            error_details["fairness_metrics"] = fairness_metrics
            
        super().__init__(message, error_details)


class DataError(BaseMLError):
    """Error raised when there is an issue with the data."""
    
    def __init__(self, reason: str, dataset_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the data error.
        
        Args:
            reason: Reason for the data error
            dataset_name: Optional name of the dataset
            details: Optional dictionary of error details
        """
        message = f"Data error: {reason}"
        
        if dataset_name:
            message = f"Data error in dataset '{dataset_name}': {reason}"
            
        error_details = details or {}
        
        if dataset_name:
            error_details["dataset_name"] = dataset_name
            
        super().__init__(message, error_details)


class FeatureEngineeringError(BaseMLError):
    """Error raised when feature engineering fails."""
    
    def __init__(self, reason: str, feature_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature engineering error.
        
        Args:
            reason: Reason for the feature engineering failure
            feature_name: Optional name of the feature
            details: Optional dictionary of error details
        """
        message = f"Feature engineering failed: {reason}"
        
        if feature_name:
            message = f"Feature engineering failed for '{feature_name}': {reason}"
            
        error_details = details or {}
        
        if feature_name:
            error_details["feature_name"] = feature_name
            
        super().__init__(message, error_details)


class DriftDetectionError(BaseMLError):
    """Error raised when drift detection fails."""
    
    def __init__(self, reason: str, monitor_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the drift detection error.
        
        Args:
            reason: Reason for the drift detection failure
            monitor_name: Optional name of the monitor
            details: Optional dictionary of error details
        """
        message = f"Drift detection failed: {reason}"
        
        if monitor_name:
            message = f"Drift detection failed for monitor '{monitor_name}': {reason}"
            
        error_details = details or {}
        
        if monitor_name:
            error_details["monitor_name"] = monitor_name
            
        super().__init__(message, error_details)


class ABTestError(BaseMLError):
    """Error raised when A/B testing fails."""
    
    def __init__(self, reason: str, test_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the A/B test error.
        
        Args:
            reason: Reason for the A/B test failure
            test_id: Optional ID of the A/B test
            details: Optional dictionary of error details
        """
        message = f"A/B test failed: {reason}"
        
        if test_id:
            message = f"A/B test '{test_id}' failed: {reason}"
            
        error_details = details or {}
        
        if test_id:
            error_details["test_id"] = test_id
            
        super().__init__(message, error_details)


class ResourceExhaustedError(BaseMLError):
    """Error raised when a resource is exhausted."""
    
    def __init__(self, resource_type: str, reason: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the resource exhausted error.
        
        Args:
            resource_type: Type of resource that was exhausted
            reason: Reason for the resource exhaustion
            details: Optional dictionary of error details
        """
        message = f"{resource_type} exhausted: {reason}"
        error_details = details or {}
        error_details["resource_type"] = resource_type
        error_details["reason"] = reason
        super().__init__(message, error_details)


class ConfigurationError(BaseMLError):
    """Error raised when there is an issue with the configuration."""
    
    def __init__(self, reason: str, config_key: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the configuration error.
        
        Args:
            reason: Reason for the configuration error
            config_key: Optional key in the configuration that caused the error
            details: Optional dictionary of error details
        """
        message = f"Configuration error: {reason}"
        
        if config_key:
            message = f"Configuration error for '{config_key}': {reason}"
            
        error_details = details or {}
        
        if config_key:
            error_details["config_key"] = config_key
            
        super().__init__(message, error_details)


class SecurityError(BaseMLError):
    """Error raised when there is a security issue."""
    
    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the security error.
        
        Args:
            reason: Reason for the security error
            details: Optional dictionary of error details
        """
        message = f"Security error: {reason}"
        super().__init__(message, details)


class CanaryTestError(BaseMLError):
    """Error raised when a canary test fails."""
    
    def __init__(self, reason: str, test_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the canary test error.
        
        Args:
            reason: Reason for the canary test failure
            test_id: Optional ID of the canary test
            details: Optional dictionary of error details
        """
        message = f"Canary test failed: {reason}"
        
        if test_id:
            message = f"Canary test '{test_id}' failed: {reason}"
            
        error_details = details or {}
        
        if test_id:
            error_details["test_id"] = test_id
            
        super().__init__(message, error_details) 