"""
ML Context Middleware

This middleware provides ML-specific context for requests in the WITHIN ML Prediction System.
It sets up ML environment information, performance monitoring, and model versioning.
"""

import time
from typing import Any, Dict, Optional
from fastapi import Request
import logging
import contextvars

# Context variable for current ML context
current_ml_context = contextvars.ContextVar('current_ml_context', default=None)


class MLPerformanceMonitor:
    """
    Performance monitoring for ML operations.
    
    This class provides a context manager for tracking ML operation performance.
    """
    
    def __init__(self):
        """Initialize the ML performance monitor."""
        self.start_time = None
        self.end_time = None
        self.processing_time_ms = None
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        """Start timing when entering the context."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Record metrics when exiting the context."""
        self.end_time = time.time()
        self.processing_time_ms = int((self.end_time - self.start_time) * 1000)
        
        # Log performance
        self.logger.debug(f"ML operation completed in {self.processing_time_ms}ms")
        
        # Add to metrics
        self.metrics["processing_time_ms"] = self.processing_time_ms
    
    def add_metric(self, name: str, value: Any):
        """
        Add a custom metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name] = value
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all recorded metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics


class MLContext:
    """
    ML context for request processing.
    
    This class provides context for ML operations during request processing.
    """
    
    def __init__(self, request_id: str, environment: str):
        """
        Initialize ML context.
        
        Args:
            request_id: Unique request identifier
            environment: ML environment (e.g., 'production', 'staging')
        """
        self.request_id = request_id
        self.environment = environment
        self.model_version = None
        self.monitor = MLPerformanceMonitor()
        self.attributes = {}
        self.logger = logging.getLogger(__name__)
    
    def set_model_version(self, model_id: str, version: str):
        """
        Set the model version for the current context.
        
        Args:
            model_id: Model identifier
            version: Model version
        """
        self.model_version = {
            "model_id": model_id,
            "version": version
        }
        self.logger.debug(f"Using model {model_id} version {version}")
    
    def add_attribute(self, name: str, value: Any):
        """
        Add a custom attribute to the context.
        
        Args:
            name: Attribute name
            value: Attribute value
        """
        self.attributes[name] = value
    
    def get_attribute(self, name: str, default: Any = None) -> Any:
        """
        Get a custom attribute from the context.
        
        Args:
            name: Attribute name
            default: Default value if attribute doesn't exist
            
        Returns:
            Attribute value or default
        """
        return self.attributes.get(name, default)


def get_ml_environment() -> str:
    """
    Get the current ML environment.
    
    Returns:
        ML environment name
    """
    # In a real implementation, this would be retrieved from configuration
    return "production"


def get_current_ml_context() -> Optional[MLContext]:
    """
    Get the current ML context from context variables.
    
    Returns:
        Current ML context or None if not set
    """
    return current_ml_context.get()


async def ml_context_middleware(request: Request, call_next):
    """
    ML context middleware.
    
    Sets up ML-specific request context with performance monitoring and environment info.
    
    Args:
        request: FastAPI request
        call_next: Next middleware/endpoint handler
        
    Returns:
        Response from next middleware/endpoint handler
    """
    # Create ML context
    request_id = request.headers.get("X-Request-ID") or str(id(request))
    ml_context = MLContext(request_id=request_id, environment=get_ml_environment())
    
    # Set context variable
    token = current_ml_context.set(ml_context)
    
    try:
        # Add ML context to request state
        request.state.ml_context = ml_context
        
        # Process the request with performance monitoring
        with ml_context.monitor:
            response = await call_next(request)
        
        # Add monitoring information to response headers
        response.headers["X-ML-Processing-Time"] = str(ml_context.monitor.processing_time_ms)
        
        # Add ML model version information if available
        if ml_context.model_version:
            response.headers["X-ML-Model-ID"] = ml_context.model_version["model_id"]
            response.headers["X-ML-Model-Version"] = ml_context.model_version["version"]
        
        return response
    
    finally:
        # Reset context variable
        current_ml_context.reset(token) 