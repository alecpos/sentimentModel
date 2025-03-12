"""
API endpoints for the WITHIN ML Prediction System.

This package contains the API endpoints for interacting with the WITHIN system,
including prediction endpoints, model management, and analytics functionality.
The API follows RESTful design principles and is versioned for compatibility.
"""

from . import v1

# API version mapping
API_VERSIONS = {
    "v1": v1
}

# Default API version
DEFAULT_API_VERSION = "v1"

__all__ = [
    "v1",
    "API_VERSIONS",
    "DEFAULT_API_VERSION"
]
