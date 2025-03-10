"""
API Response Utilities

This module provides standardized utility functions for formatting API responses
in the WITHIN ML Prediction System.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import uuid


def create_prediction_response(prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a standardized ML prediction response.
    
    Args:
        prediction_result: The prediction result from the ML service
        
    Returns:
        A standardized prediction response dictionary
    """
    # Ensure required fields exist
    if "prediction_id" not in prediction_result:
        prediction_result["prediction_id"] = str(uuid.uuid4())
        
    if "created_at" not in prediction_result:
        prediction_result["created_at"] = datetime.utcnow().isoformat()
        
    # Return the standardized response
    return prediction_result


def create_collection_response(
    items: List[Dict[str, Any]],
    total: int,
    page: int = 1,
    page_size: int = 20
) -> Dict[str, Any]:
    """
    Create a standardized paginated collection response.
    
    Args:
        items: List of items to include in the response
        total: Total number of items (across all pages)
        page: Current page number (1-indexed)
        page_size: Number of items per page
        
    Returns:
        A standardized collection response with pagination metadata
    """
    # Calculate pagination metadata
    pages = (total + page_size - 1) // page_size if page_size > 0 else 0
    has_next = page < pages
    has_prev = page > 1
    
    # Create response
    return {
        "items": items,
        "pagination": {
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": pages,
            "has_next": has_next,
            "has_prev": has_prev
        }
    }


def create_error_response(
    code: str,
    message: str,
    details: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        code: Machine-readable error code
        message: Human-readable error message
        details: Optional dictionary with field-specific validation errors
        
    Returns:
        A standardized error response dictionary
    """
    return {
        "status": "error",
        "code": code,
        "message": message,
        "details": details or {}
    }


def ml_error_responses(
    include_validation: bool = True,
    include_model_errors: bool = True,
    include_prediction_errors: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Provide OpenAPI documentation for common ML error responses.
    
    Args:
        include_validation: Whether to include validation error responses
        include_model_errors: Whether to include model-related error responses
        include_prediction_errors: Whether to include prediction-related error responses
        
    Returns:
        Dictionary mapping status codes to response examples for OpenAPI docs
    """
    responses = {}
    
    # Authentication errors
    responses[401] = {
        "description": "Authentication required",
        "content": {
            "application/json": {
                "example": create_error_response(
                    code="UNAUTHORIZED",
                    message="Authentication required"
                )
            }
        }
    }
    
    # Authorization errors
    responses[403] = {
        "description": "Insufficient permissions",
        "content": {
            "application/json": {
                "example": create_error_response(
                    code="FORBIDDEN",
                    message="Insufficient permissions to access this resource"
                )
            }
        }
    }
    
    # Validation errors
    if include_validation:
        responses[400] = {
            "description": "Invalid request parameters",
            "content": {
                "application/json": {
                    "example": create_error_response(
                        code="VALIDATION_ERROR",
                        message="Invalid request parameters",
                        details={
                            "numeric_features": ["Must contain at least 3 features"],
                            "text_features": ["Text features must be provided when image_features are absent"]
                        }
                    )
                }
            }
        }
    
    # Model errors
    if include_model_errors:
        responses[404] = {
            "description": "Model not found",
            "content": {
                "application/json": {
                    "example": create_error_response(
                        code="MODEL_NOT_FOUND",
                        message="Model with ID 'nonexistent-model' not found"
                    )
                }
            }
        }
        
        responses[500] = {
            "description": "Internal model error",
            "content": {
                "application/json": {
                    "example": create_error_response(
                        code="MODEL_ERROR",
                        message="Internal model error occurred"
                    )
                }
            }
        }
    
    # Prediction errors
    if include_prediction_errors:
        responses[422] = {
            "description": "Prediction failed",
            "content": {
                "application/json": {
                    "example": create_error_response(
                        code="PREDICTION_FAILED",
                        message="Failed to generate prediction",
                        details={
                            "reason": "Insufficient data quality",
                            "threshold": "0.8",
                            "actual": "0.65"
                        }
                    )
                }
            }
        }
        
        responses[408] = {
            "description": "Prediction timeout",
            "content": {
                "application/json": {
                    "example": create_error_response(
                        code="PREDICTION_TIMEOUT",
                        message="Prediction operation timed out"
                    )
                }
            }
        }
    
    # Rate limiting
    responses[429] = {
        "description": "Rate limit exceeded",
        "content": {
            "application/json": {
                "example": create_error_response(
                    code="TOO_MANY_REQUESTS",
                    message="Rate limit exceeded, please try again later"
                )
            }
        }
    }
    
    return responses 