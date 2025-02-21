"""Schemas for ad account health models and requests/responses."""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

class AdAccountHealthRequestSchema(BaseModel):
    """Schema for ad account health analysis request."""
    
    account_id: str = Field(..., description="Advertiser account ID")
    time_period: str = Field("last_30_days", description="Time period for analysis")
    include_metrics: List[str] = Field(default_factory=list, description="Specific metrics to include")
    comparison_period: Optional[str] = Field(None, description="Period to compare against")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "account_id": "adv789012",
                "time_period": "last_90_days",
                "include_metrics": ["engagement", "conversion", "spend_efficiency", "audience_growth"],
                "comparison_period": "previous_90_days"
            }
        }
    )

class AdAccountHealthResponseSchema(BaseModel):
    """Schema for ad account health analysis response."""
    
    id: UUID = Field(..., description="Health record ID")
    account_id: str = Field(..., description="Advertiser account ID")
    health_score: float = Field(..., description="Overall account health score (0-100)")
    engagement_trends: Dict[str, Any] = Field(..., description="Engagement trend analysis")
    risk_factors: List[Dict[str, Any]] = Field(..., description="Identified risk factors")
    optimization_suggestions: List[Dict[str, Any]] = Field(..., description="Optimization recommendations")
    spend_efficiency: float = Field(..., description="Ad spend efficiency score")
    audience_health: Dict[str, Any] = Field(..., description="Audience health metrics")
    comparison_results: Optional[Dict[str, Any]] = Field(None, description="Comparison to previous period")
    created_at: datetime = Field(..., description="Analysis timestamp")
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "e47ac10b-58cc-4372-a567-0e02b2c3d479",
                "account_id": "adv789012",
                "health_score": 78.5,
                "engagement_trends": {
                    "direction": "improving",
                    "rate_of_change": 0.12,
                    "metrics": {
                        "ctr": {"current": 0.028, "trend": 0.003},
                        "conversion_rate": {"current": 0.032, "trend": 0.005}
                    }
                },
                "risk_factors": [
                    {"type": "audience_fatigue", "severity": "medium", "affected_segments": ["returning_visitors"]},
                    {"type": "decreasing_roas", "severity": "low", "affected_campaigns": ["summer_promo"]}
                ],
                "optimization_suggestions": [
                    {"type": "audience_refresh", "impact": "high", "description": "Refresh creative for returning visitors segment"},
                    {"type": "budget_reallocation", "impact": "medium", "description": "Shift 15% budget from underperforming campaigns"}
                ],
                "spend_efficiency": 0.92,
                "audience_health": {
                    "growth_rate": 0.08,
                    "engagement_quality": "high",
                    "retention_rate": 0.76
                },
                "comparison_results": {
                    "health_score_change": 5.3,
                    "significant_improvements": ["audience_growth", "conversion_rate"],
                    "significant_declines": []
                },
                "created_at": "2025-02-20T09:30:15Z"
            }
        }
    )

class PerformanceMetricSchema(BaseModel):
    """Schema for performance metrics."""
    
    id: Optional[UUID] = Field(None, description="Metric record ID")
    account_id: str = Field(..., description="Advertiser account ID")
    metric_name: str = Field(..., description="Name of the performance metric")
    metric_value: float = Field(..., description="Value of the metric")
    time_period: str = Field(..., description="Time period (daily, weekly, monthly)")
    start_date: datetime = Field(..., description="Start of measurement period")
    end_date: datetime = Field(..., description="End of measurement period")
    is_anomaly: bool = Field(False, description="Whether this metric is flagged as anomalous")
    anomaly_score: Optional[float] = Field(None, description="Anomaly detection score if applicable")
    
    model_config = ConfigDict(from_attributes=True)