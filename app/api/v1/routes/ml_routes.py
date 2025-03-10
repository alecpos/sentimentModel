"""
ML Routes

This module registers all ML-related routes for the WITHIN ML Prediction System API.
"""

from fastapi import APIRouter
from app.api.v1.endpoints.ad_score_endpoints import router as ad_score_router

# Create main router for all ML endpoints
router = APIRouter(prefix="/api/v1")

# Include all ML-related routers
router.include_router(ad_score_router)

# Additional ML routers would be included here
# router.include_router(anomaly_router)
# router.include_router(account_health_router)
# router.include_router(model_router)
# router.include_router(prediction_router)
# router.include_router(analytics_router) 