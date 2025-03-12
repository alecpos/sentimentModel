"""
API version 1 endpoints for the WITHIN ML Prediction System.

This package defines the v1 API including:
- Ad score prediction endpoints
- Account health assessment endpoints
- Ad performance analytics
- Model management endpoints
- Authentication and authorization
"""

from fastapi import APIRouter

# Import submodules
from . import routes
from . import endpoints
from . import middleware
from . import schemas

# Create the main router
router = APIRouter(prefix="/v1", tags=["v1"])

# Include routers from route modules
router.include_router(routes.ad_score_router)
router.include_router(routes.account_health_router)
router.include_router(routes.analytics_router)
router.include_router(routes.model_management_router)

__all__ = [
    "router",
    "routes",
    "endpoints",
    "middleware",
    "schemas"
]
