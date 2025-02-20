"""WITHIN ML module for ad scoring and account health monitoring."""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID

# SQLAlchemy base for models
from sqlalchemy.ext.declarative import declarative_base
MLBase = declarative_base()

# Pydantic schemas
from .ad_score_schema import (
    AdScoreRequestSchema,
    AdScoreResponseSchema,
    AdScoreAnalysisRequestSchema,
    AdScoreAnalysisResponseSchema
)

from .ad_account_health_schema import (
    AdAccountHealthRequestSchema,
    AdAccountHealthResponseSchema,
    PerformanceMetricSchema
)

# Model version and configuration
ML_MODULE_VERSION = "1.0.0"
ML_CONFIG = {
    "feature_extraction": {
        "nlp_model": "en_core_web_lg",
        "sentiment_analyzer": "vader_lexicon",
        "min_confidence_threshold": 0.6
    },
    "scoring": {
        "model_path": "models/ad_score_v1.pkl",
        "scaler_path": "models/scaler_v1.pkl",
        "default_threshold": 0.5
    },
    "monitoring": {
        "drift_detection_window": 1000,
        "retraining_frequency_days": 14,
        "minimum_samples_for_retraining": 5000
    }
}

__all__ = [
    # Base classes
    "MLBase",
    
    # Module configuration
    "ML_MODULE_VERSION",
    "ML_CONFIG",
    
    # Ad Score Schemas
    "AdScoreRequestSchema",
    "AdScoreResponseSchema",
    "AdScoreAnalysisRequestSchema",
    "AdScoreAnalysisResponseSchema",
    
    # Ad Account Health Schemas
    "AdAccountHealthRequestSchema",
    "AdAccountHealthResponseSchema",
    "PerformanceMetricSchema"
]