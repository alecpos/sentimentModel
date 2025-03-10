# API Utilities

This directory contains utility functions and classes used by the WITHIN ML Prediction System API. These utilities provide standardized patterns for responses, error handling, validation, and other common API-related tasks.

## Overview

The utilities in this directory are designed to ensure:

1. **Consistency**: Standardized patterns across all API endpoints
2. **Reusability**: Common functionality extracted into reusable components
3. **Maintainability**: Centralized implementation of cross-cutting concerns
4. **Type Safety**: Strong typing for improved code quality

## Contents

### API Response Utilities (`api_responses.py`)

Functions and classes for standardizing API responses:

- **`create_prediction_response`**: Creates standardized ML prediction responses
- **`create_collection_response`**: Creates paginated collection responses
- **`create_error_response`**: Creates standardized error responses
- **`ml_error_responses`**: Provides OpenAPI documentation for error responses

#### Usage Example

```python
from app.utils.api_responses import create_prediction_response, ml_error_responses

@router.post("/predict", response_model=PredictionResponse, responses=ml_error_responses())
async def predict(request: PredictionRequest):
    # Process prediction
    prediction_result = await prediction_service.predict(request)
    
    # Return standardized response
    return create_prediction_response(prediction_result)
```

### ML Exceptions (`ml_exceptions.py`)

Custom exception classes for ML-specific error scenarios:

- **`MLBaseException`**: Base exception for all ML-related errors
- **`ModelNotFoundError`**: Raised when a model is not found
- **`InvalidFeatureFormatError`**: Raised when features are not in the expected format
- **`PredictionFailedError`**: Raised when a prediction fails to complete
- **`InsufficientDataError`**: Raised when there is not enough data for prediction
- **`ModelTrainingError`**: Raised when a model is training and not available
- **`ModelError`**: Raised when there is an internal model error

#### Usage Example

```python
from app.utils.ml_exceptions import ModelNotFoundError, InvalidFeatureFormatError

async def get_model(model_id: str):
    model = await model_repository.find_by_id(model_id)
    if not model:
        raise ModelNotFoundError(f"Model with ID '{model_id}' not found")
    return model

async def validate_features(features: dict):
    if not features.get("numeric_features") or len(features["numeric_features"]) < 3:
        raise InvalidFeatureFormatError(
            "Invalid feature format", 
            {"numeric_features": ["Must contain at least 3 features"]}
        )
```

### ML Validation (`ml_validation.py`)

Pydantic models and validators for ML-specific validation:

- **`BaseMLModel`**: Base Pydantic model with enhanced validation capabilities
- **`PredictionRequest`**: Base model for prediction requests
- Custom validators for feature validation

#### Usage Example

```python
from app.utils.ml_validation import BaseMLModel
from pydantic import validator, root_validator
from typing import List, Optional

class AdScorePredictionRequest(BaseMLModel):
    model_id: str = "ad-score-predictor"
    text_features: Optional[List[str]] = None
    numeric_features: Optional[List[float]] = None
    categorical_features: Optional[List[str]] = None
    include_explanation: bool = False
    
    @validator('numeric_features')
    def validate_numeric_features(cls, v):
        if v is not None and len(v) < 3:
            raise ValueError("Must contain at least 3 features")
        return v
        
    @root_validator
    def validate_has_features(cls, values):
        if not any([
            values.get('text_features'),
            values.get('numeric_features'),
            values.get('categorical_features')
        ]):
            raise ValueError("At least one feature type must be provided")
        return values
```

### Pagination Utilities (`pagination.py`)

Functions for handling API pagination:

- **`paginate`**: Applies pagination to database queries
- **`get_pagination_params`**: Extracts pagination parameters from requests
- **`create_paginated_response`**: Creates paginated responses

#### Usage Example

```python
from app.utils.pagination import get_pagination_params, paginate, create_paginated_response

@router.get("/predictions", response_model=PredictionListResponse)
async def list_predictions(request: Request):
    # Get pagination parameters
    page, page_size, sort_by, sort_order = get_pagination_params(request)
    
    # Apply pagination to query
    query = prediction_repository.list_query()
    paginated_result = await paginate(query, page, page_size, sort_by, sort_order)
    
    # Create paginated response
    return create_paginated_response(
        items=paginated_result.items,
        total=paginated_result.total,
        page=page,
        page_size=page_size
    )
```

### Authentication Utilities (`auth.py`)

Functions for handling authentication and authorization:

- **`get_current_user`**: FastAPI dependency for extracting authenticated user
- **`verify_token`**: Verifies JWT token validity
- **`create_token`**: Creates new JWT tokens
- **`require_permissions`**: Decorator for role-based permissions

#### Usage Example

```python
from app.utils.auth import get_current_user, require_permissions
from app.models.user import User

@router.get("/models/{model_id}")
@require_permissions(["models:read"])
async def get_model(model_id: str, current_user: User = Depends(get_current_user)):
    return await model_service.get_model(model_id)
```

### Rate Limiting (`rate_limit.py`)

Functions for implementing API rate limiting:

- **`rate_limiter`**: FastAPI dependency for applying rate limits
- **`configure_rate_limits`**: Configures rate limit rules
- **`track_rate_limit`**: Updates rate limit counters

#### Usage Example

```python
from app.utils.rate_limit import rate_limiter

@router.post("/predict")
async def predict(
    request: PredictionRequest,
    rate_limit: dict = Depends(rate_limiter(limit=100, period=3600))
):
    # Process prediction
    return await prediction_service.predict(request)
```

## ML-Specific Utilities

### Feature Processing (`feature_processing.py`)

Utilities for processing ML features:

- **`normalize_features`**: Normalizes numerical features
- **`encode_categorical_features`**: Encodes categorical features
- **`process_text_features`**: Processes text features
- **`prepare_features`**: Prepares features for model input

#### Usage Example

```python
from app.utils.feature_processing import prepare_features

async def predict(features: dict):
    # Prepare features for model input
    processed_features = prepare_features(features)
    
    # Use processed features for prediction
    prediction = model.predict(processed_features)
    
    return prediction
```

### Explainability (`explainability.py`)

Utilities for model explainability:

- **`generate_shap_values`**: Generates SHAP values for feature importance
- **`create_explanation`**: Creates explanation for a prediction
- **`format_feature_importance`**: Formats feature importance for API response

#### Usage Example

```python
from app.utils.explainability import create_explanation

async def predict_with_explanation(model, features):
    # Generate prediction
    prediction = model.predict(features)
    
    # Create explanation if requested
    if features.get("include_explanation"):
        explanation = create_explanation(model, features, prediction)
        prediction["explanation"] = explanation
    
    return prediction
```

## Best Practices

When using or extending these utilities:

1. **Consistency**: Follow the established patterns for new utilities
2. **Documentation**: Add comprehensive docstrings
3. **Testing**: Write unit tests for all utility functions
4. **Type Safety**: Use proper type hints for all functions and classes
5. **Error Handling**: Use appropriate exception classes
6. **Reusability**: Design for reuse across multiple endpoints 