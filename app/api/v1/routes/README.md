# API Routes Components

This directory contains route collection components for the WITHIN ML Prediction System API.

DOCUMENTATION STATUS: COMPLETE

## Purpose

The API routes system provides capabilities for:
- Organizing API endpoints into logical groupings
- Defining route hierarchies and prefixes
- Centralizing route registration
- Managing API versioning
- Routing requests to appropriate endpoint handlers

## Key Components

### ML Routes

Components for machine learning API routes:
- Ad Score prediction routes
- Account health assessment routes
- Model management routes
- Training and evaluation routes
- Feature importance routes

### Route Management

Components for route organization and management:
- Router collection and registration
- Route prefixing and tagging
- Version-specific routing
- Route dependencies and middleware
- Rate limiting and authentication requirements

## Usage Example

```python
from fastapi import FastAPI
from app.api.v1.routes import ml_router

# Create FastAPI app
app = FastAPI()

# Include all ML-related routes
app.include_router(ml_router)
```

## Integration Points

- **API Layer**: Routes connect API endpoints to business logic
- **Authentication**: Routes integrate with authentication middleware
- **ML Models**: Routes connect requests to ML prediction services
- **Data Access**: Routes provide access to data storage and retrieval
- **Monitoring**: Routes are instrumented for performance monitoring

## Dependencies

- FastAPI router system for route definition and organization
- Endpoint implementation modules that define the actual request handlers
- Authentication and authorization middleware
- Validation schemas for request/response data
- Monitoring and logging components

## Implementation Notes

The routes package follows a modular design where:
1. Each route collection (e.g., `ml_routes.py`) groups related endpoints
2. The `__init__.py` file exports routers for external use
3. The actual endpoint implementations are in the `endpoints` directory

### Fix for ad_score_router Import

This directory previously had an empty `__init__.py` file, which caused import errors when trying to access `routes.ad_score_router` from `app.api.v1.__init__.py`. 

This has been fixed by properly exporting both the main route collections and specific routers like `ad_score_router` to maintain backward compatibility. 