"""Model for storing ad scoring results."""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import uuid4

from sqlalchemy import Column, String, JSON, Float, DateTime
from sqlalchemy.orm import relationship

from app.core.database import BaseModel

class AdScoreModel(BaseModel):
    """Model for storing ad scoring results."""
    
    __tablename__ = "ad_scores"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    ad_id = Column(String, nullable=False)
    advertiser_id = Column(String, nullable=False)
    ad_content = Column(String, nullable=False)
    engagement_score = Column(Float, nullable=False)
    sentiment_score = Column(Float, nullable=False)
    complexity_score = Column(Float, nullable=False)
    topics = Column(JSON, nullable=False, default=list)
    target_audience_match = Column(Float, nullable=False)
    predicted_ctr = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    processed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Rename 'metadata' to something else, like 'meta_info'
    meta_info = Column(JSON, nullable=False, default=dict)

class AdScoreAnalysisModel(BaseModel):
    """Model for storing detailed ad score analysis results."""
    
    __tablename__ = "ad_score_analysis"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    ad_score_id = Column(String(36), nullable=False)
    content_features = Column(JSON, nullable=False, default=dict)
    linguistic_analysis = Column(JSON, nullable=False, default=dict)
    visual_elements = Column(JSON, nullable=False, default=list)
    performance_projections = Column(JSON, nullable=False, default=dict)
    similar_successful_ads = Column(JSON, nullable=False, default=list)
    improvement_suggestions = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)