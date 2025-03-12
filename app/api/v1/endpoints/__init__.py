"""
API endpoints for the WITHIN ML Prediction System.

This module defines FastAPI endpoint implementations for the ML Prediction System,
providing the actual request handlers that process API requests and return responses.
It includes endpoints for ad score prediction, account health assessment, data catalog
management, and other system functionalities.

Key functionality includes:
- Ad score prediction endpoints
- Account health assessment endpoints
- Data catalog management endpoints
- Model metadata endpoints
- Health and status endpoints

DOCUMENTATION STATUS: COMPLETE
"""

__version__ = "0.1.0"  # Pre-release version

# Export router from endpoint modules
from .ad_score_endpoints import router as ad_score_router

# Export base classes
from .base import BaseEndpoint, BaseMLEndpoint

__all__ = [
    "ad_score_router",
    "BaseEndpoint",
    "BaseMLEndpoint"
] 