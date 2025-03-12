"""
ML system database models for the WITHIN ML Prediction System.

This module provides SQLAlchemy ORM models for ML system operation, tracking,
and management. These models handle the persistence of ML metadata, versioning,
training jobs, and prediction logs.

Key entities include:
- Model metadata and versions
- Training jobs and history
- Prediction logs and results
- Feature importance metrics
- Model performance tracking

DOCUMENTATION STATUS: COMPLETE
"""

__version__ = "0.1.0"  # Pre-release version

# Import base models
from app.core.db import BaseModel, FullModel, TimestampedModel

# Placeholder for future ML system model implementations
# from .model_metadata_model import ModelMetadata
# from .training_job_model import TrainingJob
# from .prediction_log_model import PredictionLog
# from .feature_importance_model import FeatureImportance
# from .performance_metrics_model import ModelPerformanceMetrics

# Export models
__all__ = [
    # List of model classes to export
    # "ModelMetadata",
    # "TrainingJob",
    # "PredictionLog",
    # "FeatureImportance",
    # "ModelPerformanceMetrics"
] 