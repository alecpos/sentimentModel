"""
Database models for the WITHIN ML Prediction System.

This module provides SQLAlchemy ORM models that represent the database schema
for the ML Prediction System. These models are used for persistence, database
operations, and maintaining relational integrity across the system.

Key functionality includes:
- Data persistence models
- Relationship definitions
- Database schema representation
- ORM query interfaces
- Data validation rules

DOCUMENTATION STATUS: COMPLETE
"""

__version__ = "0.1.0"  # Pre-release version

# Import base model and utilities
from app.core.db import BaseModel, TimestampedModel, FullModel

# Import models from subdirectories (commented until models are implemented)
# from . import users
# from . import campaigns
# from . import ml_system
# from . import analytics

# Reexport key models from subdirectories
# These will be populated as models are implemented

__all__ = [
    # Subdirectories (will be uncommented as implemented)
    # "users",
    # "campaigns",
    # "ml_system",
    # "analytics",
    
    # Key models from subdirectories (will be populated as implemented)
    # Users
    # "User",
    # "Role",
    
    # Campaigns
    # "Campaign",
    # "Ad",
    # "PerformanceMetrics",
    
    # ML System
    # "ModelMetadata",
    # "PredictionLog",
    
    # Analytics
    # "Report",
    # "Dashboard"
] 