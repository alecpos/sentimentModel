"""Model for storing ad account health metrics."""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import uuid4

from sqlalchemy import Column, String, JSON, Float, DateTime, Boolean
from sqlalchemy.orm import relationship

from app.core.database import BaseModel

class AdAccountHealthModel(BaseModel):
    """Model for storing ad account health metrics."""
    
    __tablename__ = "ad_account_health"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    account_id = Column(String, nullable=False)
    health_score = Column(Float, nullable=False)
    engagement_trends = Column(JSON, nullable=False)
    risk_factors = Column(JSON, nullable=False, default=list)
    optimization_suggestions = Column(JSON, nullable=False, default=list)
    historical_performance = Column(JSON, nullable=False)
    spend_efficiency = Column(Float, nullable=False)
    audience_health = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

class PerformanceMetricModel(BaseModel):
    """Model for storing time-series performance metrics."""
    
    __tablename__ = "performance_metrics"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    account_id = Column(String, nullable=False)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    time_period = Column(String, nullable=False)  # daily, weekly, monthly
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    is_anomaly = Column(Boolean, default=False)
    anomaly_score = Column(Float, nullable=True)
    meta_info = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)