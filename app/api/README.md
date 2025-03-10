# API Documentation

This directory contains the API endpoints and route definitions for the WITHIN Ad Score & Account Health Predictor system. The API is built using FastAPI and follows RESTful design principles.

## API Structure

The API is organized into logical groupings by functionality:

```
api/
├── __init__.py                # API package initialization
├── routes/                    # Route definitions by domain
│   ├── __init__.py            # Routes package initialization
│   ├── ad_score.py            # Ad score prediction endpoints
│   ├── account_health.py      # Account health endpoints
│   ├── analytics.py           # Analytics data endpoints
│   └── auth.py                # Authentication endpoints
├── dependencies.py            # Shared API dependencies
├── middleware.py              # API middleware (logging, error handling, etc.)
├── models/                    # API data models (Pydantic)
│   ├── __init__.py            # Models package initialization
│   ├── ad_score.py            # Ad score request/response models
│   ├── account_health.py      # Account health request/response models
│   └── common.py              # Shared data models
└── utils/                     # API-specific utilities
    ├── __init__.py            # Utils package initialization
    ├── rate_limiting.py       # Rate limiting implementation
    ├── validation.py          # Input validation utilities
    └── response.py            # Response formatting utilities
```

## Core Endpoints

### Ad Score Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/ad-score/predict` | POST | Generate ad effectiveness score |
| `/api/v1/ad-score/batch-predict` | POST | Generate scores for multiple ads |
| `/api/v1/ad-score/explain/{prediction_id}` | GET | Get explanation for a prediction |
| `/api/v1/ad-score/history` | GET | Get prediction history |

### Account Health Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/account-health/score` | GET | Get current account health score |
| `/api/v1/account-health/trends` | GET | Get account health trends over time |
| `/api/v1/account-health/risks` | GET | Get identified risk factors |
| `/api/v1/account-health/recommendations` | GET | Get account improvement recommendations |

### Analytics Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/analytics/ad-performance` | GET | Get ad performance analytics |
| `/api/v1/analytics/platform-comparison` | GET | Compare performance across platforms |
| `/api/v1/analytics/audience-insights` | GET | Get audience demographic insights |

## Authentication

All API endpoints require authentication using JWT tokens, except for the authentication endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/auth/login` | POST | Authenticate and receive a token |
| `/api/v1/auth/refresh` | POST | Refresh an expired token |
| `/api/v1/auth/logout` | POST | Invalidate the current token |

## Request Examples

### Ad Score Prediction

```python
import requests
import json

url = "https://your-instance.within.com/api/v1/ad-score/predict"
headers = {
    "Authorization": "Bearer YOUR_TOKEN",
    "Content-Type": "application/json"
}
payload = {
    "ad_content": {
        "headline": "Limited Time Offer: 20% Off All Products",
        "description": "Shop our entire collection and save with this exclusive discount.",
        "cta": "Shop Now"
    },
    "platform": "facebook",
    "target_audience": ["fashion_shoppers", "deal_seekers"],
    "historical_metrics": {
        "avg_ctr": 0.025,
        "avg_conversion_rate": 0.03
    }
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print(response.json())
```

Example response:

```json
{
    "prediction_id": "ad_score_123456",
    "score": 78.5,
    "confidence": 0.92,
    "score_components": {
        "content_quality": 82.3,
        "audience_match": 75.8,
        "historical_performance": 79.2
    },
    "timestamp": "2023-04-15T10:30:45Z",
    "model_version": "2.1.0"
}
```

### Account Health Score

```python
import requests

url = "https://your-instance.within.com/api/v1/account-health/score"
headers = {
    "Authorization": "Bearer YOUR_TOKEN"
}
params = {
    "account_id": "123456789",
    "platform": "google"
}

response = requests.get(url, headers=headers, params=params)
print(response.json())
```

Example response:

```json
{
    "account_id": "123456789",
    "platform": "google",
    "health_score": 85,
    "component_scores": {
        "performance": 88,
        "efficiency": 82,
        "growth": 75,
        "stability": 92
    },
    "trend": "+3.5 from previous month",
    "timestamp": "2023-04-15T10:30:45Z"
}
```

## Error Handling

The API uses standard HTTP status codes and returns error responses in a consistent format:

```json
{
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid input data",
        "details": [
            {
                "field": "ad_content.headline",
                "issue": "exceeds_max_length",
                "max_length": 100
            }
        ]
    },
    "request_id": "req_abc123"
}
```

Common error codes:

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | VALIDATION_ERROR | Invalid input data |
| 401 | UNAUTHORIZED | Missing or invalid credentials |
| 403 | FORBIDDEN | Insufficient permissions |
| 404 | NOT_FOUND | Resource not found |
| 429 | RATE_LIMITED | Too many requests |
| 500 | SERVER_ERROR | Internal server error |

## Rate Limiting

The API implements rate limiting to ensure fair usage. Limits are applied per API key:

| Endpoint Type | Rate Limit |
|---------------|------------|
| Prediction endpoints | 100 requests per minute |
| Batch prediction endpoints | 10 requests per minute |
| Analytics endpoints | 30 requests per minute |
| Account health endpoints | 30 requests per minute |

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1618498245
```

## Versioning

The API uses URI versioning (e.g., `/api/v1/`). When breaking changes are introduced, a new version will be released while maintaining support for previous versions according to the deprecation schedule.

## Performance Metrics

Performance metrics are included in response headers:

```
X-Process-Time: 0.235
X-Model-Version: 2.1.0
X-Cache-Hit: true
```

## Documentation

Interactive OpenAPI documentation is available at:
- Swagger UI: `/docs`
- ReDoc: `/redoc`

## Dependencies

- **FastAPI**: Web framework
- **Pydantic**: Data validation
- **PyJWT**: JWT token handling
- **Uvicorn**: ASGI server

## Development Guidelines

When developing new API endpoints:

1. Follow RESTful design principles
2. Use appropriate HTTP methods (GET, POST, PUT, DELETE)
3. Implement proper status codes and error formats
4. Include request validation
5. Add OpenAPI/Swagger documentation
6. Implement caching for frequent predictions
7. Add monitoring for latency and throughput
8. Use appropriate batch processing
9. Implement rate limiting
10. Implement proper authentication
11. Add input sanitization
12. Include audit logging
13. Apply appropriate access controls