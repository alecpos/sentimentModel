"""
API middleware components for the WITHIN ML Prediction System.

This module provides middleware components for the FastAPI application, handling
cross-cutting concerns such as authentication, request logging, error handling,
and performance monitoring across API endpoints.

Key functionality includes:
- Authentication and authorization middleware
- Request/response logging and monitoring
- Rate limiting and throttling
- Error handling and standardized responses
- Performance tracking and metrics collection
"""

__version__ = "0.1.0"  # Pre-release version

# Package-level constants
DEFAULT_RATE_LIMIT = 100  # requests per minute
DEFAULT_TIMEOUT = 30  # seconds
LOG_REQUEST_BODY = False  # Whether to log request bodies by default

# When implementations are added, they will be imported and exported here 