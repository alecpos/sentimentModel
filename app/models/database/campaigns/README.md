# Campaign Database Models

This directory contains SQLAlchemy database models for ad campaign management in the WITHIN ML Prediction System.

DOCUMENTATION STATUS: COMPLETE

## Purpose

The campaign models system provides database representations for:
- Managing ad campaign data and metadata
- Tracking ad creative content and performance
- Handling campaign settings across different platforms
- Recording budget and spending information
- Storing performance metrics and history

## Future Components

This directory is prepared for implementing the following database models:

### Campaign Model

Core campaign entity representation:
- Campaign identification and metadata
- Status and lifecycle tracking
- Campaign objectives and goals
- Targeting settings
- Date ranges and scheduling

### Ad Creative Models

Ad content and creative models:
- Ad content and media references
- Creative variations
- Copy and message text
- Visual components
- Call-to-action elements

### Platform Settings Models

Platform-specific configuration:
- Platform identifiers and references
- Platform-specific settings
- API connection details
- Platform constraints
- Delivery settings

### Budget and Spending Models

Financial tracking models:
- Budget allocations
- Spending limits and caps
- Cost tracking
- ROI calculations
- Budget utilization metrics

### Performance Metrics Models

Campaign performance data:
- Impression and click metrics
- Conversion tracking
- Engagement metrics
- Cost metrics (CPC, CPM, etc.)
- Historical performance data

## Usage Example

Once implemented, campaign models would be used like this:

```python
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.database.campaigns import Campaign, Ad, PerformanceMetrics
from app.core.db import get_db

def get_campaign_performance(campaign_id: str, days: int = 30):
    db = next(get_db())
    
    # Get the campaign
    campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
    
    if not campaign:
        return None
    
    # Get performance metrics for the last N days
    start_date = datetime.utcnow() - timedelta(days=days)
    
    metrics = db.query(PerformanceMetrics)\
               .filter(PerformanceMetrics.campaign_id == campaign_id)\
               .filter(PerformanceMetrics.date >= start_date)\
               .order_by(PerformanceMetrics.date)\
               .all()
    
    # Get all ads for this campaign
    ads = db.query(Ad).filter(Ad.campaign_id == campaign_id).all()
    
    return {
        "campaign": campaign.to_dict(),
        "ads": [ad.to_dict() for ad in ads],
        "metrics": [metric.to_dict() for metric in metrics],
        "summary": {
            "total_impressions": sum(m.impressions for m in metrics),
            "total_clicks": sum(m.clicks for m in metrics),
            "total_conversions": sum(m.conversions for m in metrics),
            "total_spend": sum(m.spend for m in metrics),
            "avg_ctr": sum(m.clicks for m in metrics) / sum(m.impressions for m in metrics) if sum(m.impressions for m in metrics) > 0 else 0,
            "avg_cpc": sum(m.spend for m in metrics) / sum(m.clicks for m in metrics) if sum(m.clicks for m in metrics) > 0 else 0
        }
    }
```

## Integration Points

- **Campaign Management API**: CRUD operations on campaign data
- **Ad Score Prediction**: Campaign data used for ML feature extraction
- **Budget Optimization**: Campaign performance used for budget allocation
- **Reporting System**: Campaign data used for report generation
- **External Platform APIs**: Campaign settings synced with external platforms

## Dependencies

- SQLAlchemy ORM for database operations
- Core DB components (FullModel with audit trails)
- JSON serialization for complex settings
- Validation utilities for constraint checking
- Date/time utilities for scheduling 