"""Schemas for ad score models and requests/responses."""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

class AdScoreRequestSchema(BaseModel):
    """Schema for ad score request."""
    
    ad_id: str = Field(..., description="Unique identifier for the ad")
    advertiser_id: str = Field(..., description="ID of the advertiser")
    ad_content: str = Field(..., description="Full ad content text")
    ad_type: str = Field(..., description="Type of ad (image, video, carousel, etc.)")
    target_audience: Optional[List[str]] = Field(None, description="Target audience segments")
    platform: str = Field(..., description="Platform where ad will be displayed")
    historical_performance: Optional[Dict[str, Any]] = Field(None, description="Historical performance data if available")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ad_id": "ad123456",
                "advertiser_id": "adv789012",
                "ad_content": "Experience the future of fitness with TechFit Pro. Advanced workout tracking, personalized coaching, and real-time feedback.",
                "ad_type": "image",
                "target_audience": ["fitness_enthusiasts", "tech_savvy", "25-40_age_group"],
                "platform": "instagram",
                "historical_performance": {
                    "avg_ctr": 0.025,
                    "avg_conversion_rate": 0.03,
                    "previous_engagement_rate": 0.045
                }
            }
        }
    )

class AdScoreResponseSchema(BaseModel):
    """Schema for ad score response."""
    
    id: UUID = Field(..., description="Unique identifier for the ad score record")
    ad_id: str = Field(..., description="Ad identifier")
    engagement_score: float = Field(..., description="Overall predicted engagement score (0-1)")
    sentiment_score: float = Field(..., description="Sentiment analysis score (-1 to 1)")
    complexity_score: float = Field(..., description="Content complexity score (0-10)")
    topics: List[str] = Field(..., description="Main topics identified in the ad")
    target_audience_match: float = Field(..., description="Match score with target audience (0-1)")
    predicted_ctr: float = Field(..., description="Predicted click-through rate")
    confidence_score: float = Field(..., description="Model confidence in prediction (0-1)")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Suggestions to improve ad performance")
    processed_at: datetime = Field(..., description="Timestamp when the ad was scored")
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                "ad_id": "ad123456",
                "engagement_score": 0.78,
                "sentiment_score": 0.65,
                "complexity_score": 3.2,
                "topics": ["fitness", "technology", "health", "innovation"],
                "target_audience_match": 0.82,
                "predicted_ctr": 0.032,
                "confidence_score": 0.89,
                "improvement_suggestions": [
                    "Include a stronger call-to-action",
                    "Emphasize cost savings more prominently",
                    "Add social proof elements"
                ],
                "processed_at": "2025-02-20T10:15:30Z"
            }
        }
    )

class AdScoreAnalysisRequestSchema(BaseModel):
    """Schema for requesting detailed ad score analysis."""
    
    ad_score_id: UUID = Field(..., description="ID of the previously scored ad")
    analysis_depth: str = Field("standard", description="Depth of analysis: 'basic', 'standard', or 'comprehensive'")
    include_similar_ads: bool = Field(True, description="Whether to include similar successful ads")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ad_score_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                "analysis_depth": "comprehensive",
                "include_similar_ads": True
            }
        }
    )

class AdScoreAnalysisResponseSchema(BaseModel):
    """Schema for detailed ad score analysis response."""
    
    id: UUID = Field(..., description="Analysis record ID")
    ad_score_id: UUID = Field(..., description="Referenced ad score ID")
    content_features: Dict[str, Any] = Field(..., description="Extracted content features")
    linguistic_analysis: Dict[str, Any] = Field(..., description="Linguistic analysis results")
    visual_elements: List[Dict[str, Any]] = Field(..., description="Visual element analysis if applicable")
    performance_projections: Dict[str, Any] = Field(..., description="Projected performance metrics")
    similar_successful_ads: List[Dict[str, Any]] = Field(..., description="Similar high-performing ads")
    improvement_suggestions: List[Dict[str, Any]] = Field(..., description="Detailed improvement suggestions")
    created_at: datetime = Field(..., description="Timestamp of analysis")
    
    model_config = ConfigDict(from_attributes=True)