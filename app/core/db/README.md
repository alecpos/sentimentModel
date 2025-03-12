# Database Core Components

This directory contains database core components for the WITHIN ML Prediction System.

DOCUMENTATION STATUS: COMPLETE

## Purpose

The database core system provides capabilities for:
- Managing database connections and sessions
- Defining base models and common model behaviors
- Standardizing ORM patterns across the application
- Implementing reusable database mixins for common functionality
- Facilitating database operations with helper utilities

## Key Components

### Base Components

Core database setup and configuration:
- SQLAlchemy engine configuration
- Session management
- Connection pooling
- Table creation utilities
- Database context management

### Base Models

Foundational model classes:
- Abstract base model with common functionality
- Automatic table name generation
- Dictionary conversion utilities
- Model serialization/deserialization
- Query builders and common operations

### Model Mixins

Reusable components for database models:
- `TimestampMixin`: Created and updated timestamps
- `UUIDMixin`: UUID primary key generation
- `SoftDeleteMixin`: Soft delete functionality
- `AuditMixin`: Audit trail for changes (who created/updated)
- `VersioningMixin`: Version history tracking

### Composite Models

Pre-configured model bases combining multiple mixins:
- `TimestampedModel`: Base model with timestamps
- `FullModel`: Complete model with UUID, timestamps, soft delete, and audit

## Usage Example

```python
from app.core.db import Base, FullModel, get_db, engine
from sqlalchemy import Column, String, Float, Integer

# Define a model using provided base classes
class Campaign(FullModel):
    """Campaign data model."""
    name = Column(String(255), nullable=False, index=True)
    budget = Column(Float, nullable=False)
    platform = Column(String(50), nullable=False)
    impressions = Column(Integer, default=0)
    
    # Custom methods can be added
    def is_active(self):
        return self.deleted_at is None and self.budget > 0

# Create tables
Base.metadata.create_all(bind=engine)

# Use the database session
def get_campaigns():
    db = next(get_db())
    campaigns = db.query(Campaign).filter(Campaign.is_deleted.is_(False)).all()
    return [campaign.to_dict() for campaign in campaigns]
```

## Integration Points

- **Models**: Domain models extend these base components
- **Repositories**: Data access code uses these session utilities
- **API Layer**: Endpoints use database sessions for persistence
- **Services**: Business logic interacts with models
- **Migrations**: Database schema migrations build on these base models

## Dependencies

- SQLAlchemy ORM for database operations
- Database driver for your specific database (SQLite included by default)
- Python datetime and UUID libraries for automatic field generation
- JSON serialization for versioning and complex data 