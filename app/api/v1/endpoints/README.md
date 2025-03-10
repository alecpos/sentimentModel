# API Endpoints

This directory contains the endpoint controllers for the WITHIN ML Prediction System API v1. Each file in this directory implements the controller logic for a specific API domain.

## Architecture

The endpoint controllers follow a consistent architecture:

1. **Controller Functions**: Handle HTTP requests and format responses
2. **Service Integration**: Delegate business logic to service layer
3. **Input Validation**: Validate request data using Pydantic schemas
4. **Error Handling**: Use standardized error handling patterns
5. **Response Formatting**: Format responses consistently

## Endpoint Controllers

### Ad Score Prediction (`ad_score_endpoints.py`)

Endpoints for predicting ad performance scores:

- **`predict_ad_score`**: Creates predictions for ad performance
- **`get_ad_score_features`**: Returns supported features
- **`get_ad_score_metrics`**: Returns model performance metrics

### Anomaly Detection (`anomaly_endpoints.py`)

Endpoints for detecting anomalies in advertising data:

- **`detect_anomalies`**: Detects anomalies in ad performance data
- **`batch_detect_anomalies`**: Performs batch anomaly detection
- **`get_anomaly_types`**: Returns supported anomaly types

### Account Health (`account_health_endpoints.py`)

Endpoints for assessing account health:

- **`predict_account_health`**: Predicts account health score
- **`get_health_factors`**: Returns health factors
- **`get_health_recommendations`**: Returns improvement recommendations

### Models (`model_endpoints.py`)

Endpoints for ML model management:

- **`list_models`**: Lists available ML models
- **`get_model`**: Gets model information
- **`list_model_versions`**: Lists model versions
- **`get_model_version`**: Gets specific model version
- **`get_model_metrics`**: Gets model performance metrics

### Predictions (`prediction_endpoints.py`)

Endpoints for managing predictions:

- **`create_prediction`**: Creates a new prediction
- **`batch_predict`**: Creates batch predictions
- **`get_prediction`**: Gets prediction details
- **`list_predictions`**: Lists historical predictions

### Analytics (`analytics_endpoints.py`)

Endpoints for analytics and explainability:

- **`get_model_performance`**: Gets model performance analytics
- **`get_feature_importance`**: Gets feature importance analysis
- **`explain_prediction`**: Gets explanation for a prediction
- **`list_explanation_methods`**: Lists available explanation methods

## Implementation Patterns

Each endpoint controller follows these implementation patterns:

### Request Processing

```python
@router.post("/ad-score/predict", response_model=AdScorePredictionResponse, 
             responses=ml_error_responses())
async def predict_ad_score(
    request: AdScorePredictionRequest,
    current_user: User = Depends(get_current_user)
) -> AdScorePredictionResponse:
    """
    Predict ad performance score based on provided features.
    
    Args:
        request: The prediction request containing features
        current_user: The authenticated user
        
    Returns:
        The prediction result with score, confidence, and explanations
        
    Raises:
        InvalidFeatureFormatError: Features are not in the expected format
        PredictionFailedError: The prediction failed to complete
    """
    try:
        # Delegate to service layer
        prediction = await ad_score_service.predict(
            request.text_features,
            request.numeric_features,
            request.categorical_features,
            request.image_features,
            include_explanation=request.include_explanation
        )
        
        # Format response
        return create_prediction_response(prediction)
        
    except InvalidFeatureFormatError as e:
        # Handle validation errors
        raise HTTPException(
            status_code=400,
            detail=create_error_response(
                code="INVALID_FEATURE_FORMAT",
                message="Features are not in the expected format",
                details=e.details
            ).dict()
        )
    except PredictionFailedError as e:
        # Handle prediction errors
        raise HTTPException(
            status_code=422,
            detail=create_error_response(
                code="PREDICTION_FAILED",
                message="Failed to generate prediction",
                details=e.details
            ).dict()
        )
```

### Pagination Implementation

```python
@router.get("/predictions", response_model=PredictionListResponse)
async def list_predictions(
    request: Request,
    current_user: User = Depends(get_current_user)
) -> PredictionListResponse:
    """
    List historical predictions.
    
    Args:
        request: The HTTP request containing pagination parameters
        current_user: The authenticated user
        
    Returns:
        Paginated list of predictions
    """
    # Get pagination parameters
    page, page_size, sort_by, sort_order = get_pagination_params(request)
    
    # Get filtered predictions
    predictions, total = await prediction_service.list_predictions(
        user_id=current_user.id,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order
    )
    
    # Create paginated response
    return create_collection_response(
        items=predictions,
        total=total,
        page=page,
        page_size=page_size
    )
```

### Error Handling

All endpoint controllers use standardized error handling with ML-specific error codes:

```python
try:
    # Operation that might fail
    result = await some_operation()
    return result
except ModelNotFoundError as e:
    raise HTTPException(
        status_code=404,
        detail=create_error_response(
            code="MODEL_NOT_FOUND",
            message=str(e)
        ).dict()
    )
except InvalidInputError as e:
    raise HTTPException(
        status_code=400,
        detail=create_error_response(
            code="VALIDATION_ERROR",
            message="Invalid input parameters",
            details=e.details
        ).dict()
    )
except Exception as e:
    # Log unexpected errors
    logger.error(f"Unexpected error: {str(e)}")
    raise HTTPException(
        status_code=500,
        detail=create_error_response(
            code="INTERNAL_ERROR",
            message="An unexpected error occurred"
        ).dict()
    )
```

## Adding New Endpoints

When adding new endpoints:

1. **Follow Existing Patterns**: Use the patterns demonstrated in existing controllers
2. **Document Thoroughly**: Include comprehensive docstrings
3. **Validate All Inputs**: Use Pydantic schemas for validation
4. **Handle Errors Consistently**: Use standardized error handling
5. **Test Thoroughly**: Write unit and integration tests for all endpoints

## Testing

Each endpoint controller has corresponding tests in the `tests/api/v1/endpoints` directory. These tests cover:

- Valid requests and responses
- Input validation
- Error conditions
- Edge cases

## Security Considerations

All endpoints implement:

- **Authentication**: JWT token validation
- **Authorization**: Role-based permissions
- **Input Validation**: Strict schema validation
- **Rate Limiting**: Protection against abuse 

class BaseMLEndpoint:
    """Base class for ML-related endpoints with standardized patterns."""
    
    def __init__(self, service):
        self.service = service
    
    async def handle_prediction_request(self, request, user):
        """Handle ML prediction requests with standardized error handling."""
        try:
            # Delegate to service layer
            result = await self.service.predict(request, user)
            
            # Format response
            return create_prediction_response(result)
        except InvalidFeatureFormatError as e:
            # Handle validation errors
            raise self._format_validation_error(e)
        except PredictionFailedError as e:
            # Handle prediction errors
            raise self._format_prediction_error(e)
    
    def _format_validation_error(self, error):
        """Format validation error according to API standards."""
        return HTTPException(
            status_code=400,
            detail=create_error_response(
                code="INVALID_FEATURE_FORMAT",
                message="Features are not in the expected format",
                details=error.details
            ).dict()
        )
    
    def _format_prediction_error(self, error):
        """Format prediction error according to API standards."""
        return HTTPException(
            status_code=422,
            detail=create_error_response(
                code="PREDICTION_FAILED",
                message="Failed to generate prediction",
                details=error.details
            ).dict()
        ) 