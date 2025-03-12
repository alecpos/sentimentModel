# Database Models

This directory contains SQLAlchemy database models for the WITHIN ML Prediction System.

DOCUMENTATION STATUS: COMPLETE

## Purpose

The database models system provides capabilities for:
- Defining the database schema as Python classes
- Representing entity relationships
- Enforcing data integrity constraints
- Supporting object-relational mapping (ORM)
- Facilitating database queries and operations

## Directory Structure

This module is organized into subdirectories by domain:

### users/

Models for user management and access control:
- User accounts and profiles
- Authentication credentials
- Role and permission assignments
- Access tokens and sessions
- Audit logs for security events

### campaigns/

Models for ad campaign data:
- Campaigns and campaign metadata
- Ad creative content and metrics
- Platform-specific campaign settings
- Campaign performance history
- Budget and spending records

### ml_system/

Models for ML system operation:
- Model metadata and versioning
- Training job records
- Prediction logs and history
- Model performance metrics
- Feature importance tracking

### analytics/

Models for analytics and reporting:
- Report configurations and templates
- Saved queries and visualizations
- Scheduled report definitions
- Dashboard configurations
- Export and sharing settings

## Usage Example

Once implemented, database models would be used like this:

```python
from sqlalchemy.orm import Session
from app.models.database.campaigns import Campaign, Ad
from app.models.database.users import User
from app.core.db import get_db

def get_user_campaigns(user_id: str):
    db = next(get_db())
    
    # Get the user
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        return None
    
    # Get campaigns owned by this user
    campaigns = db.query(Campaign)\
                  .filter(Campaign.owner_id == user.id)\
                  .order_by(Campaign.created_at.desc())\
                  .all()
    
    # Get ads for each campaign
    result = []
    for campaign in campaigns:
        ads = db.query(Ad).filter(Ad.campaign_id == campaign.id).all()
        
        result.append({
            "campaign": campaign.to_dict(),
            "ads": [ad.to_dict() for ad in ads]
        })
    
    return result
```

## Integration Points

- **Data Access Layer**: Repositories use these models for CRUD operations
- **API Layer**: Endpoints convert between these models and API schemas
- **Business Logic**: Services use these models for domain operations
- **ML Services**: ML components reference these models for entity relationships
- **Authentication**: Auth system uses user and permission models

## Dependencies

- SQLAlchemy ORM for object-relational mapping
- Core DB components (BaseModel, mixins, etc.)
- Database connection and session management
- Validation utilities for data integrity
- Type definitions and annotations 