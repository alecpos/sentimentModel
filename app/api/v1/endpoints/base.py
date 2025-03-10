"""
Base Endpoint Classes

This module provides base classes for API endpoints in the WITHIN ML Prediction System.
These classes implement standardized patterns for handling requests, responses, and errors.
"""

from typing import Any, Dict, List, Optional, Type, Union
from fastapi import HTTPException, Depends, Request
from pydantic import BaseModel

from app.utils.api_responses import create_prediction_response, create_error_response
from app.utils.ml_exceptions import (
    MLBaseException,
    ModelNotFoundError,
    InvalidFeatureFormatError,
    PredictionFailedError,
    InsufficientDataError,
    ModelError
)


class BaseEndpoint:
    """Base class for all endpoints with standardized patterns."""
    
    def __init__(self, service):
        """
        Initialize base endpoint.
        
        Args:
            service: Service class for business logic
        """
        self.service = service
    
    def _format_error(self, 
                      error: Exception, 
                      status_code: int = 500, 
                      code: str = "INTERNAL_ERROR", 
                      message: str = "An unexpected error occurred") -> HTTPException:
        """
        Format error as HTTPException with standardized structure.
        
        Args:
            error: The exception to format
            status_code: HTTP status code
            code: Error code
            message: Error message
            
        Returns:
            HTTPException with formatted error
        """
        return HTTPException(
            status_code=status_code,
            detail=create_error_response(
                code=code,
                message=message
            ).dict()
        )


class BaseMLEndpoint(BaseEndpoint):
    """Base class for ML-related endpoints with standardized patterns."""
    
    async def handle_prediction_request(self, request: Any, user: Any = None) -> Dict[str, Any]:
        """
        Handle ML prediction requests with standardized error handling.
        
        Args:
            request: The prediction request
            user: Authenticated user
            
        Returns:
            Standardized prediction response
            
        Raises:
            HTTPException: On error with standardized format
        """
        try:
            # Delegate to service layer
            result = await self.service.predict(request, user)
            
            # Format response
            return create_prediction_response(result)
        except InvalidFeatureFormatError as e:
            # Handle validation errors
            raise self._format_validation_error(e)
        except PredictionFailedError as e:
            # Handle prediction errors
            raise self._format_prediction_error(e)
        except ModelNotFoundError as e:
            # Handle model not found errors
            raise self._format_model_not_found_error(e)
        except InsufficientDataError as e:
            # Handle insufficient data errors
            raise self._format_insufficient_data_error(e)
        except ModelError as e:
            # Handle model errors
            raise self._format_model_error(e)
        except Exception as e:
            # Handle unexpected errors
            import logging
            logging.getLogger(__name__).exception(f"Unexpected error during prediction: {str(e)}")
            raise self._format_error(e)
    
    def _format_validation_error(self, error: InvalidFeatureFormatError) -> HTTPException:
        """
        Format validation error according to API standards.
        
        Args:
            error: The validation error
            
        Returns:
            HTTPException with formatted error
        """
        return HTTPException(
            status_code=400,
            detail=create_error_response(
                code="INVALID_FEATURE_FORMAT",
                message="Features are not in the expected format",
                details=error.details
            ).dict()
        )
    
    def _format_prediction_error(self, error: PredictionFailedError) -> HTTPException:
        """
        Format prediction error according to API standards.
        
        Args:
            error: The prediction error
            
        Returns:
            HTTPException with formatted error
        """
        return HTTPException(
            status_code=422,
            detail=create_error_response(
                code="PREDICTION_FAILED",
                message="Failed to generate prediction",
                details=error.details
            ).dict()
        )
    
    def _format_model_not_found_error(self, error: ModelNotFoundError) -> HTTPException:
        """
        Format model not found error according to API standards.
        
        Args:
            error: The model not found error
            
        Returns:
            HTTPException with formatted error
        """
        return HTTPException(
            status_code=404,
            detail=create_error_response(
                code="MODEL_NOT_FOUND",
                message=str(error),
                details=error.details
            ).dict()
        )
    
    def _format_insufficient_data_error(self, error: InsufficientDataError) -> HTTPException:
        """
        Format insufficient data error according to API standards.
        
        Args:
            error: The insufficient data error
            
        Returns:
            HTTPException with formatted error
        """
        return HTTPException(
            status_code=400,
            detail=create_error_response(
                code="INSUFFICIENT_DATA",
                message=str(error),
                details=error.details
            ).dict()
        )
    
    def _format_model_error(self, error: ModelError) -> HTTPException:
        """
        Format model error according to API standards.
        
        Args:
            error: The model error
            
        Returns:
            HTTPException with formatted error
        """
        return HTTPException(
            status_code=500,
            detail=create_error_response(
                code="MODEL_ERROR",
                message=str(error),
                details=error.details
            ).dict()
        ) 