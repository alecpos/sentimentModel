# API v1 Documentation

This directory contains the version 1 implementation of the WITHIN ML Prediction System API. The API provides standardized endpoints for machine learning model management, predictions, and analytics.

## Base URL

All API v1 endpoints are prefixed with `/api/v1`.

## Authentication

All API endpoints require authentication using JSON Web Tokens (JWT).

### Authentication Headers

```
Authorization: Bearer <token>
```

## API Endpoints

### ML Model Management

| Method | Endpoint                            | Description                           |
|--------|-------------------------------------|---------------------------------------|
| GET    | /models                             | List available ML models              |
| GET    | /models/{model_id}                  | Get model information                 |
| GET    | /models/{model_id}/versions         | List model versions                   |
| GET    | /models/{model_id}/versions/{version_id} | Get specific model version      |
| GET    | /models/{model_id}/metrics          | Get model performance metrics         |

### Predictions

| Method | Endpoint                            | Description                           |
|--------|-------------------------------------|---------------------------------------|
| POST   | /predict                            | Create a new prediction               |
| POST   | /batch-predict                      | Create batch predictions              |
| GET    | /predictions/{prediction_id}        | Get prediction details                |
| GET    | /predictions                        | List historical predictions           |

### Ad Score Prediction

| Method | Endpoint                            | Description                           |
|--------|-------------------------------------|---------------------------------------|
| POST   | /ad-score/predict                   | Predict ad performance score          |
| GET    | /ad-score/features                  | Get supported features                |
| GET    | /ad-score/metrics                   | Get model performance metrics         |

### Anomaly Detection

| Method | Endpoint                            | Description                           |
|--------|-------------------------------------|---------------------------------------|
| POST   | /anomaly/detect                     | Detect anomalies in ad data           |
| POST   | /anomaly/detect-batch               | Batch anomaly detection               |
| GET    | /anomaly/types                      | Get supported anomaly types           |

### Account Health

| Method | Endpoint                            | Description                           |
|--------|-------------------------------------|---------------------------------------|
| POST   | /account-health/predict             | Predict account health                |
| GET    | /account-health/factors             | Get health factors                    |
| GET    | /account-health/recommendations     | Get improvement recommendations       |

### Analytics and Explainability

| Method | Endpoint                            | Description                           |
|--------|-------------------------------------|---------------------------------------|
| GET    | /analytics/model-performance        | Get model performance analytics       |
| GET    | /analytics/feature-importance       | Get feature importance analysis       |
| POST   | /explain                            | Get explanation for a prediction      |
| GET    | /explain/methods                    | List available explanation methods    |

## Request and Response Examples

### Ad Score Prediction

#### Request

```http
POST /api/v1/ad-score/predict
Content-Type: application/json
Authorization: Bearer <token>

{
  "text_features": ["Compelling ad copy with call to action"],
  "numeric_features": [0.1, 0.5, 0.3, 0.7],
  "categorical_features": ["fashion", "mobile"],
  "image_features": null,
  "include_explanation": true
}
```

#### Response

```json
{
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
```

### Anomaly Detection

#### Request

```http
POST /api/v1/anomaly/detect
Content-Type: application/json
Authorization: Bearer <token>

{
  "ad_performance_data": [
    {
      "date": "2023-06-01",
      "impressions": 10000,
      "clicks": 350,
      "conversions": 15
    },
    {
      "date": "2023-06-02",
      "impressions": 12000,
      "clicks": 420,
      "conversions": 18
    },
    {
      "date": "2023-06-03",
      "impressions": 9000,
      "clicks": 50,
      "conversions": 2
    }
  ],
  "sensitivity": "medium"
}
```

#### Response

```json
{
  "detection_id": "234e5678-e89b-12d3-a456-426614174000",
  "model_id": "anomaly-detector",
  "model_version": "1.3.0",
  "anomalies": [
    {
      "date": "2023-06-03",
      "metrics": ["clicks", "conversions"],
      "severity": "high",
      "expected_values": {
        "clicks": 315,
        "conversions": 13.5
      },
      "actual_values": {
        "clicks": 50,
        "conversions": 2
      },
      "explanation": "Click-through rate dropped by 85% compared to previous trend."
    }
  ],
  "processing_time_ms": 87,
  "created_at": "2023-06-10T14:32:00Z"
}
```

### Account Health Assessment

#### Request

```http
POST /api/v1/account-health/predict
Content-Type: application/json
Authorization: Bearer <token>

{
  "account_id": "acc-12345",
  "performance_history": [
    {
      "period": "2023-05",
      "impressions": 1500000,
      "clicks": 75000,
      "conversions": 3750,
      "spend": 15000
    },
    {
      "period": "2023-06",
      "impressions": 1400000,
      "clicks": 65000,
      "conversions": 2900,
      "spend": 15500
    }
  ],
  "include_recommendations": true
}
```

#### Response

```json
{
  "assessment_id": "345e6789-e89b-12d3-a456-426614174000",
  "model_id": "account-health-predictor",
  "model_version": "1.8.0",
  "health_score": 0.68,
  "risk_level": "medium",
  "key_indicators": {
    "performance_trend": "declining",
    "efficiency": "stable",
    "conversion_quality": "deteriorating"
  },
  "recommendations": [
    {
      "issue": "Declining conversion rate",
      "recommendation": "Review landing page experience",
      "impact": "high",
      "effort": "medium"
    },
    {
      "issue": "Rising cost per conversion",
      "recommendation": "Optimize bidding strategy",
      "impact": "medium",
      "effort": "low"
    }
  ],
  "processing_time_ms": 210,
  "created_at": "2023-06-10T14:35:00Z"
}
```

### Feature Importance Analysis

#### Request

```http
GET /api/v1/analytics/feature-importance?model_id=ad-score-predictor
Authorization: Bearer <token>
```

#### Response

```json
{
  "model_id": "ad-score-predictor",
  "model_version": "2.5.0",
  "feature_importance": {
    "text_sentiment": 0.35,
    "image_quality": 0.25,
    "ad_relevance": 0.20,
    "historical_ctr": 0.15,
    "audience_match": 0.05
  },
  "methodology": "shap_values",
  "sample_size": 10000,
  "created_at": "2023-06-10T00:00:00Z"
}
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages in the response body.

### Error Response Format

```json
{
  "status": "error",
  "code": "ERROR_CODE",
  "message": "A detailed error message",
  "details": {
    "field1": ["Error details related to field1"],
    "field2": ["Error details related to field2"]
  }
}
```

### Common Error Codes

| Status Code | Error Code           | Description                                     |
|-------------|----------------------|-------------------------------------------------|
| 400         | VALIDATION_ERROR     | Invalid request parameters                      |
| 400         | INVALID_FEATURE_FORMAT | Features are not in the expected format       |
| 401         | UNAUTHORIZED         | Authentication required                         |
| 403         | FORBIDDEN            | Insufficient permissions                        |
| 404         | MODEL_NOT_FOUND      | The requested model doesn't exist               |
| 404         | NOT_FOUND            | Resource not found                              |
| 408         | PREDICTION_TIMEOUT   | The prediction operation timed out              |
| 422         | PREDICTION_FAILED    | The prediction failed to complete               |
| 429         | TOO_MANY_REQUESTS    | Rate limit exceeded                             |
| 500         | MODEL_ERROR          | Internal model error                            |
| 500         | INTERNAL_ERROR       | Server error                                    |

### Error Examples

#### Invalid Features

```json
{
  "status": "error",
  "code": "INVALID_FEATURE_FORMAT",
  "message": "Features are not in the expected format",
  "details": {
    "numeric_features": ["Must contain at least 3 features"],
    "text_features": ["Text features must be provided when image_features are absent"]
  }
}
```

#### Model Not Found

```json
{
  "status": "error",
  "code": "MODEL_NOT_FOUND",
  "message": "Model with ID 'nonexistent-model' not found",
  "details": {}
}
```

#### Prediction Failed

```json
{
  "status": "error",
  "code": "PREDICTION_FAILED",
  "message": "Failed to generate prediction",
  "details": {
    "reason": "Insufficient data quality",
    "threshold": "0.8",
    "actual": "0.65"
  }
}
```

## Pagination

List endpoints support pagination with the following query parameters:

- `page`: Page number (1-based, default: 1)
- `page_size`: Number of items per page (default: 20, max: 100)
- `sort_by`: Field to sort by (depends on the endpoint)
- `sort_order`: Sort order (`asc` or `desc`, default: `asc`)

Paginated responses include the following metadata:

```json
{
  "items": [...],
  "pagination": {
    "total": 100,
    "page": 1,
    "page_size": 20,
    "pages": 5,
    "has_next": true,
    "has_prev": false
  }
}
```

## Rate Limiting

The API implements rate limiting to protect against abuse and ensure fair usage. Rate limits vary by endpoint, with prediction endpoints having stricter limits due to their computational cost.

Rate limit headers are included in the response:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1680278400
```

## Schema Definitions

API request and response schemas are defined in the `schemas` directory using Pydantic models. These models enforce validation and provide serialization/deserialization capabilities.

## Implementation Details

The implementation of the API endpoints is organized as follows:

- **routes/**: Route definitions that map HTTP methods to controller functions
- **endpoints/**: Controller functions that handle requests and format responses
- **schemas/**: Pydantic models for request/response validation and serialization 