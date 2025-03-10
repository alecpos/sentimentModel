# API Schemas

This directory contains Pydantic models for request and response validation in the WITHIN ML Prediction System API. These schemas define the structure, validation rules, and documentation for API inputs and outputs.

## Overview

The schemas serve multiple purposes:

1. **Input Validation**: Validate incoming request data
2. **Response Serialization**: Define the structure of API responses
3. **Documentation**: Generate OpenAPI/Swagger documentation
4. **Type Safety**: Provide type hints for improved code quality

## Schema Categories

### Base Models (`base_models.py`)

Base Pydantic models that are extended by other schemas:

- **`BaseModel`**: Enhanced Pydantic BaseModel with common configuration
- **`BaseMLModel`**: Base model for ML-related requests with additional validation
- **`PaginationParams`**: Common pagination parameters
- **`PaginatedResponse`**: Base model for paginated responses

### Request Models

Models for validating API request data:

- **`ad_score_schemas.py`**: Ad score prediction requests
- **`anomaly_schemas.py`**: Anomaly detection requests
- **`account_health_schemas.py`**: Account health assessment requests
- **`model_schemas.py`**: Model management requests
- **`prediction_schemas.py`**: Generic prediction requests
- **`analytics_schemas.py`**: Analytics and explainability requests

### Response Models

Models for structuring API responses:

- **`ad_score_schemas.py`**: Ad score prediction responses
- **`anomaly_schemas.py`**: Anomaly detection responses
- **`account_health_schemas.py`**: Account health assessment responses
- **`model_schemas.py`**: Model management responses
- **`prediction_schemas.py`**: Generic prediction responses
- **`analytics_schemas.py`**: Analytics and explainability responses

### Error Models (`error_schemas.py`)

Models for standardized error responses:

- **`ErrorResponse`**: Standard error response structure
- **`ValidationErrorResponse`**: Detailed validation error response
- **`MLErrorResponse`**: ML-specific error response

## Validation Patterns

### Feature Validation

The schemas implement comprehensive validation for ML features:

```python
class AdScorePredictionRequest(BaseMLModel):
    """Request for ad score prediction."""
    
    model_id: str = "ad-score-predictor"
    text_features: Optional[List[str]] = None
    numeric_features: Optional[List[float]] = None
    categorical_features: Optional[List[str]] = None
    image_features: Optional[List[str]] = None
    include_explanation: bool = False
    
    @validator('numeric_features')
    def validate_numeric_features(cls, v):
        """Validate numeric features."""
        if v is not None and len(v) < 3:
            raise ValueError("Must contain at least 3 features")
        return v
        
    @validator('text_features')
    def validate_text_features(cls, v, values):
        """Validate text features."""
        if v is not None and any(len(text) > 1000 for text in v):
            raise ValueError("Text features cannot exceed 1000 characters")
        return v
    
    @root_validator
    def validate_has_features(cls, values):
        """Validate that at least one feature type is provided."""
        has_features = any([
            values.get('text_features'),
            values.get('numeric_features'),
            values.get('categorical_features'),
            values.get('image_features')
        ])
        
        if not has_features:
            raise ValueError("At least one feature type must be provided")
        return values
```

### Advanced Validation Examples

The schemas include advanced validation patterns such as:

#### Dependent Field Validation

```python
class ScheduleOptimizationRequest(BaseMLModel):
    """Request for schedule optimization."""
    
    optimization_type: str
    start_date: datetime
    end_date: datetime
    include_weekends: bool = False
    priority_tasks: Optional[List[str]] = None
    
    @validator('end_date')
    def validate_end_date(cls, v, values):
        """Validate that end_date is after start_date."""
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError("end_date must be after start_date")
        return v
    
    @validator('priority_tasks')
    def validate_priority_tasks(cls, v, values):
        """Validate priority tasks based on optimization type."""
        if values.get('optimization_type') == "focused" and not v:
            raise ValueError("priority_tasks must be provided for focused optimization")
        return v
```

#### List Validation

```python
class BatchPredictionRequest(BaseMLModel):
    """Request for batch predictions."""
    
    model_id: str
    items: List[Dict[str, Any]]
    
    @validator('items')
    def validate_items(cls, v):
        """Validate items list."""
        if not v:
            raise ValueError("items cannot be empty")
        
        if len(v) > 100:
            raise ValueError("Maximum of 100 items can be processed in batch")
            
        return v
```

#### Enum Validation

```python
class AnomalyDetectionRequest(BaseMLModel):
    """Request for anomaly detection."""
    
    ad_performance_data: List[Dict[str, Any]]
    sensitivity: Literal["low", "medium", "high"] = "medium"
    
    @validator('sensitivity')
    def validate_sensitivity(cls, v):
        """Validate sensitivity value."""
        valid_values = ["low", "medium", "high"]
        if v not in valid_values:
            raise ValueError(f"sensitivity must be one of {valid_values}")
        return v
```

## Response Models

Response models define the structure of API responses:

```python
class AdScorePredictionResponse(BaseModel):
    """Response for ad score prediction."""
    
    prediction_id: str
    model_id: str
    model_version: str
    score: float
    confidence: float
    processing_time_ms: int
    created_at: datetime
    explanation: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "prediction_id": "123e4567-e89b-12d3-a456-426614174000",
                "model_id": "ad-score-predictor",
                "model_version": "2.5.0",
                "score": 0.87,
                "confidence": 0.92,
                "processing_time_ms": 125,
                "created_at": "2023-06-10T14:30:00Z",
                "explanation": {
                    "method": "shap",
                    "feature_importance": {
                        "text_features[0]": 0.45,
                        "numeric_features[2]": 0.30,
                        "categorical_features[0]": 0.25
                    },
                    "baseline_score": 0.50
                }
            }
        }
```

## Pagination Models

Models for handling paginated responses:

```python
class PaginationMetadata(BaseModel):
    """Pagination metadata for collection responses."""
    
    total: int
    page: int
    page_size: int
    pages: int
    has_next: bool
    has_prev: bool

class PredictionListResponse(BaseModel):
    """Paginated response for prediction listings."""
    
    items: List[PredictionSummary]
    pagination: PaginationMetadata
```

## Error Models

Models for standardized error responses:

```python
class ErrorResponse(BaseModel):
    """Standardized error response."""
    
    status: str = "error"
    code: str
    message: str
    details: Dict[str, List[str]] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "status": "error",
                "code": "VALIDATION_ERROR",
                "message": "Invalid request parameters",
                "details": {
                    "numeric_features": ["Must contain at least 3 features"],
                    "text_features": ["Text features must be provided when image_features are absent"]
                }
            }
        }
```

## Best Practices

When working with these schemas:

1. **Reuse Base Models**: Extend from base models for consistency
2. **Document Thoroughly**: Include docstrings, examples, and field descriptions
3. **Validate Comprehensively**: Implement appropriate validators
4. **Keep Models Focused**: Each model should have a single responsibility
5. **Include Examples**: Provide examples for OpenAPI documentation
6. **Use Type Annotations**: Include proper typing for all fields 