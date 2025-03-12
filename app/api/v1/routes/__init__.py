"""
API Routes for the WITHIN ML Prediction System.

This module exports all API route collections for the v1 API.

DOCUMENTATION STATUS: COMPLETE
"""

from app.api.v1.routes.ml_routes import router as ml_router
from app.api.v1.endpoints.ad_score_endpoints import router as ad_score_router

# For backward compatibility, also export ad_score_router directly
# This ensures that app.api.v1.__init__.py can still access routes.ad_score_router

__all__ = [
    "ml_router",
    "ad_score_router",
]
