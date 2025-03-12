# Schemas Directory

This directory contains Pydantic schema definitions for the WITHIN ML Prediction System. These schemas define the structure and validation rules for data used throughout the application, particularly for API request and response models.

## Directory Structure

- **__init__.py**: Module initialization with schema exports
- **ad_score_schema.py**: Schemas for ad score prediction
- **ad_account_health_schema.py**: Schemas for account health assessment

## Key Components

### Ad Score Schemas

Located in `ad_score_schema.py`, these schemas define the structure of ad score prediction data:

- **AdScoreRequest**: Input data for ad score prediction
- **AdScoreResponse**: Response format for ad score predictions
- **AdScoreFeatures**: Feature definitions for ad scoring
- **AdScoreMetrics**: Performance metrics for ad scores

### Account Health Schemas

Located in `ad_account_health_schema.py`, these schemas define the structure of account health assessment data:

- **AccountHealthRequest**: Input data for account health assessment
- **AccountHealthResponse**: Response format for account health assessments
- **AccountHealthMetrics**: Metrics for account health evaluation
- **AccountRiskFactors**: Risk factors that affect account health

## Usage Examples

### Validating Request Data

```python
from app.schemas import AdScoreRequest

def process_ad_score_request(raw_data: dict):
    # Validate and parse the incoming data
    try:
        validated_data = AdScoreRequest(**raw_data)
        return validated_data
    except ValueError as e:
        raise ValidationError(f"Invalid request data: {str(e)}")
```

### Creating Response Objects

```python
from app.schemas import AdScoreResponse
from datetime import datetime

def create_ad_score_response(ad_id: str, score: float, confidence: float):
    return AdScoreResponse(
        ad_id=ad_id,
        score=score,
        confidence=confidence,
        timestamp=datetime.now().isoformat(),
        model_version="1.0.0"
    )
```

## Design Principles

The schema definitions follow these principles:

1. **Strict Validation**: All schemas implement strict validation rules
2. **Self-Documentation**: Schemas include detailed field descriptions
3. **Type Safety**: All fields have explicit type annotations
4. **Version Compatibility**: Schemas maintain backward compatibility
5. **Extensibility**: Schemas can be extended for future requirements

## Dependencies

- **Pydantic**: For schema definition and validation
- **Python typing**: For type annotations
- **datetime**: For timestamp handling
- **enum**: For enumeration types

## Additional Resources

- See [Pydantic documentation](https://docs.pydantic.dev/) for schema definition details
- See `app/api/README.md` for API endpoint information that uses these schemas 