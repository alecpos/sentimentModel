# API Overview

**IMPLEMENTATION STATUS: IMPLEMENTED**


The WITHIN Ad Score & Account Health Predictor system provides a comprehensive REST API that allows users to interact with the prediction models, access analytics, and manage data. This document provides an overview of the API architecture, design principles, and usage patterns.

## API Design Principles

The WITHIN API follows these core principles:

1. **RESTful Design**: Clear resource-oriented URL structure with appropriate HTTP methods
2. **Consistent Responses**: Standardized response formats and error handling
3. **Versioned Endpoints**: Clear versioning to ensure compatibility
4. **Comprehensive Documentation**: OpenAPI/Swagger documentation for all endpoints
5. **Authentication & Authorization**: Secure access controls for all API resources
6. **Performance**: Fast response times with appropriate caching
7. **Observability**: Detailed logging and monitoring of API usage

## API Base URL

```
https://api.within.co/api/v1
```

All API endpoints are accessible under this base URL. The `v1` indicates the API version, which will increment for breaking changes.

## API Architectural Overview

The WITHIN API is organized around the following components:

```
API
 ├── Authentication
 │   ├── Login
 │   ├── Token Refresh
 │   └── Logout
 │
 ├── Ad Score Prediction
 │   ├── Single Prediction
 │   ├── Batch Prediction
 │   ├── Prediction Explanation
 │   └── Prediction History
 │
 ├── Account Health
 │   ├── Health Score
 │   ├── Health Trends
 │   ├── Risk Factors
 │   └── Recommendations
 │
 ├── Analytics
 │   ├── Ad Performance
 │   ├── Platform Comparison
 │   └── Audience Insights
 │
 └── Data Management
     ├── Ad Data
     ├── Account Data
     └── Platform Integration
```

## Authentication

The API uses JSON Web Tokens (JWT) for authentication. To access protected endpoints, you must:

1. Obtain an access token using your credentials
2. Include the token in the `Authorization` header of subsequent requests
3. Refresh the token when it expires

See the [Authentication Guide](authentication.md) for detailed instructions.

## Core API Resources

### Ad Score Prediction

The ad score prediction API allows you to:

- Generate effectiveness scores for ad content
- Predict performance across different platforms
- Understand what factors influence scores
- Track prediction history over time

Example endpoint: `POST /ad-score/predict`

### Account Health Assessment

The account health API allows you to:

- Get an overall health score for an advertising account
- Track health trends over time
- Identify risk factors affecting performance
- Get recommendations for improvement

Example endpoint: `GET /account-health/score?account_id={id}&platform={platform}`

### Analytics

The analytics API provides access to:

- Ad performance metrics and trends
- Comparisons across platforms
- Audience insights and segmentation
- Custom report generation

Example endpoint: `GET /analytics/ad-performance?ad_id={id}&time_range=last_30_days`

## Request Formats

The API accepts inputs in the following formats:

### JSON Requests

Most endpoints accept JSON-formatted request bodies:

```json
{
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
```

### Query Parameters

Some endpoints, especially for retrieval operations, use query parameters:

```
/account-health/trends?account_id=123456789&platform=google&time_range=last_90_days
```

### Path Parameters

Resources that operate on specific items use path parameters:

```
/ad-score/explain/ad_score_123456
```

## Response Format

All API responses follow a consistent format:

### Success Responses

```json
{
  "data": {
    // Resource-specific response data
  },
  "meta": {
    "request_id": "req_abc123",
    "timestamp": "2023-04-15T10:30:45Z",
    "processing_time_ms": 235
  }
}
```

### Error Responses

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
  "meta": {
    "request_id": "req_abc123",
    "timestamp": "2023-04-15T10:30:45Z"
  }
}
```

## Pagination

For endpoints that return collections of items, the API supports pagination:

```
/analytics/ad-performance?limit=20&offset=40
```

Paginated responses include metadata:

```json
{
  "data": [
    // Items
  ],
  "meta": {
    "pagination": {
      "total_count": 157,
      "offset": 40,
      "limit": 20,
      "next_offset": 60,
      "previous_offset": 20
    },
    "request_id": "req_abc123",
    "timestamp": "2023-04-15T10:30:45Z"
  }
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage. Rate limits are applied per API key and vary by endpoint type:

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

When a rate limit is exceeded, the API returns a `429 Too Many Requests` status code.

## Performance Metrics

Performance metrics are included in response headers:

```
X-Process-Time: 0.235
X-Model-Version: 2.1.0
X-Cache-Hit: true
```

## Supported Content Types

The API supports the following content types:

- `application/json` (default)
- `multipart/form-data` (for file uploads)

Specify the desired response format using the `Accept` header.

## SDK Support

Official client libraries are available for:

- Python
- JavaScript (Node.js and browser)
- Java
- PHP

See the [SDK Documentation](../development/sdks.md) for installation and usage instructions.

## Common Use Cases

### Predicting Ad Effectiveness

```python
import requests

url = "https://api.within.co/api/v1/ad-score/predict"
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

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

### Monitoring Account Health

```python
import requests

url = "https://api.within.co/api/v1/account-health/score"
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

## Related Documentation

- [API Endpoints Reference](endpoints.md): Detailed documentation of all API endpoints
- [Authentication Guide](authentication.md): Guide to authentication and authorization
- [API Usage Examples](examples.md): Common usage patterns and examples
- [API Error Codes](error_codes.md): Reference for all error codes and messages
- [API SDKs](../development/sdks.md): Official client libraries for various programming languages

## API Versioning

The API uses URI versioning (e.g., `/api/v1/`). When breaking changes are introduced, a new version will be released while maintaining support for previous versions according to the deprecation schedule:

- **Minor Changes**: Backward-compatible changes may be added to existing versions
- **Major Changes**: Breaking changes trigger a new API version
- **Deprecation**: Old versions are maintained for at least 12 months after deprecation
- **Communication**: API changes are announced at least 3 months before deprecation 