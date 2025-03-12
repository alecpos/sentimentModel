"""
Campaign-related database models for the WITHIN ML Prediction System.

This module provides SQLAlchemy ORM models for ad campaign management, tracking,
and analysis. These models handle the persistence of campaign data and related entities.

Key entities include:
- Campaigns and campaign metadata
- Ad creatives and content
- Platform-specific campaign settings
- Campaign performance metrics
- Budget and spending tracking

DOCUMENTATION STATUS: COMPLETE
"""

__version__ = "0.1.0"  # Pre-release version

# Import base models
from app.core.db import BaseModel, FullModel, TimestampedModel

# Placeholder for future campaign model implementations
# from .campaign_model import Campaign
# from .ad_model import Ad
# from .budget_model import Budget
# from .performance_model import PerformanceMetrics

# Export models
__all__ = [
    # List of model classes to export
    # "Campaign",
    # "Ad",
    # "Budget",
    # "PerformanceMetrics"
] 