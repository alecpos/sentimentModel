# ML Inference API Documentation

**IMPLEMENTATION STATUS: PARTIALLY_IMPLEMENTED**


## Overview

This document provides detailed specifications for the ML Inference API, which serves predictions from the WITHIN machine learning models. The API follows RESTful design principles and includes endpoints for ad scoring, account health assessment, sentiment analysis, and related ML-powered capabilities.

## Table of Contents

1. [API Principles](#api-principles)
2. [Authentication](#authentication)
3. [Endpoints](#endpoints)
4. [Request/Response Formats](#request-response-formats)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Versioning](#versioning)
8. [Performance Characteristics](#performance-characteristics)
9. [Integration Examples](#integration-examples)
10. [Monitoring and Debugging](#monitoring-and-debugging)

## API Principles

The ML Inference API follows these core principles:

1. **Stateless**: Each request contains all necessary information
2. **Idempotent**: Repeated identical requests produce identical results
3. **Predictable**: Well-defined behavior for all input variations
4. **Secure**: Authenticated and authorized access only
5. **Observable**: Comprehensive logging and monitoring
6. **Resilient**: Graceful degradation under load
7. **Scalable**: Horizontal scaling for high throughput

## Authentication

Authentication uses JWT tokens with the following characteristics:

- **Header Format**: `Authorization: Bearer <token>`
- **Token Lifetime**: 1 hour
- **Refresh Flow**: Use `/auth/refresh` endpoint with refresh token
- **Scope-Based Access**: Different scopes for different ML capabilities
- **Rate Limits**: Tied to authentication level

Example authentication:

```python
import requests

API_URL = "https://api.within.co/v1/ml"
API_KEY = "your_api_key"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

response = requests.post(
    f"{API_URL}/ad-score",
    headers=headers,
    json={"headline": "Amazing offer inside!", "description": "Limited time discount"}
)
```

## Endpoints

### Ad Score Prediction

**Endpoint**: `/ad-score`  
**Method**: POST  
**Description**: Evaluates ad effectiveness on a 0-100 scale

**Request Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| headline | string | Yes | Ad headline text |
| description | string | No | Ad description text |
| cta | string | No | Call-to-action text |
| image_url | string | No | URL to ad image |
| platform | string | No | Target platform (facebook, google, etc.) |
| industry | string | No | Industry vertical |
| additional_attributes | object | No | Platform-specific attributes |

**Response Format**:

```json
{
  "score": 85,
  "confidence": 0.92,
  "category": "high",
  "explanations": [
    {"factor": "headline_effectiveness", "impact": 0.4, "description": "Strong emotional appeal in headline"},
    {"factor": "description_clarity", "impact": 0.3, "description": "Clear value proposition in description"},
    {"factor": "cta_strength", "impact": 0.2, "description": "Direct and action-oriented CTA"}
  ],
  "recommendations": [
    {"description": "Consider adding social proof elements", "estimated_impact": "+5 points"},
    {"description": "Try a more specific CTA", "estimated_impact": "+3 points"}
  ],
  "request_id": "req_ad7f9e2c13",
  "model_version": "2.1.0"
}
```

### Account Health Assessment

**Endpoint**: `/account-health`  
**Method**: POST  
**Description**: Analyzes overall health of an advertising account

**Request Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| account_id | string | Yes | Platform account identifier |
| platform | string | Yes | Advertising platform name |
| time_range | string | No | Analysis time range (default: last_30_days) |
| include_recommendations | boolean | No | Whether to include recommendations (default: true) |
| metrics | array | No | Specific metrics to analyze |

**Response Format**:

```json
{
  "health_score": 72,
  "trend": "stable",
  "risk_level": "medium",
  "issues": [
    {
      "category": "budget_allocation",
      "description": "Budget distribution across campaigns is suboptimal",
      "impact": "high",
      "affected_entities": ["campaign_123", "campaign_456"]
    },
    {
      "category": "targeting_overlap",
      "description": "Significant audience overlap between campaigns",
      "impact": "medium",
      "affected_entities": ["campaign_123", "campaign_789"]
    }
  ],
  "recommendations": [
    {
      "description": "Redistribute budget from low to high performing campaigns",
      "potential_impact": "+15% ROAS",
      "confidence": 0.85,
      "difficulty": "medium"
    },
    {
      "description": "Refine audience targeting to reduce overlap",
      "potential_impact": "+8% CTR",
      "confidence": 0.78,
      "difficulty": "high"
    }
  ],
  "metrics": {
    "overall_roas": 3.2,
    "trend_direction": "positive",
    "anomalies_detected": 2
  },
  "request_id": "req_bf82e41d98",
  "model_version": "1.8.5"
}
```

### Ad Sentiment Analysis

**Endpoint**: `/ad-sentiment`  
**Method**: POST  
**Description**: Analyzes sentiment and emotional tone of ad content

**Request Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| text | string | Yes | Ad text to analyze |
| detailed | boolean | No | Whether to return detailed analysis (default: false) |

**Response Format**:

```json
{
  "sentiment": "positive",
  "sentiment_score": 0.78,
  "confidence": 0.92,
  "emotions": {
    "joy": 0.64,
    "trust": 0.55,
    "anticipation": 0.48,
    "surprise": 0.22,
    "anger": 0.05,
    "fear": 0.03,
    "disgust": 0.02,
    "sadness": 0.01
  },
  "key_phrases": [
    {"text": "limited offer", "salience": 0.85},
    {"text": "exclusive access", "salience": 0.72},
    {"text": "special discount", "salience": 0.67}
  ],
  "tone_attributes": {
    "formal": 0.28,
    "persuasive": 0.82,
    "urgent": 0.75,
    "informative": 0.45
  },
  "request_id": "req_2f7a13c4d9",
  "model_version": "1.3.2"
}
```

### Batch Prediction

**Endpoint**: `/batch-predict`  
**Method**: POST  
**Description**: Processes multiple items in a single request

**Request Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| model | string | Yes | Model identifier (ad-score, account-health, sentiment) |
| items | array | Yes | Array of items to process |
| options | object | No | Model-specific options |

**Response Format**:

```json
{
  "results": [
    {
      "item_id": "item_1",
      "prediction": { /* model-specific prediction result */ },
      "status": "success"
    },
    {
      "item_id": "item_2",
      "prediction": { /* model-specific prediction result */ },
      "status": "success"
    },
    {
      "item_id": "item_3",
      "error": "Invalid input: missing required field",
      "status": "error"
    }
  ],
  "summary": {
    "total": 3,
    "successful": 2,
    "failed": 1
  },
  "request_id": "req_9c73b2a1e5",
  "model_version": "2.0.1"
}
```

## Request/Response Formats

### Content Types

- Request: `application/json`
- Response: `application/json`

### Response Structure

All API responses follow this structure:

```json
{
  // Model-specific result fields
  ...,
  
  // Metadata fields (always present)
  "request_id": "string",
  "model_version": "string",
  "processing_time": "number (ms)"
}
```

### Date Formats

All dates use ISO 8601 format (`YYYY-MM-DDTHH:MM:SSZ`).

## Error Handling

The API uses HTTP status codes and a consistent error response format:

```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": [
      {
        "field": "string",
        "description": "string"
      }
    ]
  },
  "request_id": "string"
}
```

### HTTP Status Codes

| Status Code | Description | Recovery Strategy |
|-------------|-------------|------------------|
| 400 | Bad Request - Invalid input parameters | Fix the request parameters according to the error details |
| 401 | Unauthorized - Missing or invalid authentication | Refresh authentication token or check API credentials |
| 403 | Forbidden - Insufficient permissions | Request access to the required scope or resource |
| 404 | Not Found - Endpoint or resource doesn't exist | Verify the endpoint URL or resource identifier |
| 422 | Unprocessable Entity - Valid request but processing failed | Check error details for specific validation failures |
| 429 | Too Many Requests - Rate limit exceeded | Implement exponential backoff and respect rate limits |
| 500 | Internal Server Error - Server-side error | Retry with exponential backoff; contact support if persistent |
| 503 | Service Unavailable - Temporary unavailability | Retry after the period specified in Retry-After header |

### Error Codes

The `code` field in error responses provides specific error types:

| Error Code | Description | Example | Remediation |
|------------|-------------|---------|------------|
| `invalid_parameter` | One or more parameters are invalid | Missing required field or invalid format | Review parameters against documentation |
| `validation_failed` | Input validation checks failed | Text too long, invalid date range | Adjust inputs to meet validation criteria |
| `model_error` | ML model processing error | Insufficient features for prediction | Provide additional context or features |
| `resource_not_found` | Referenced resource doesn't exist | Account ID not found | Verify resource identifiers |
| `rate_limit_exceeded` | API rate limit reached | Too many requests per minute | Implement rate limiting in client code |
| `service_unavailable` | Temporary service disruption | Scheduled maintenance | Retry with backoff strategy |
| `feature_not_enabled` | Feature not available for account | Beta features not enabled | Request feature enablement |
| `data_quality_issue` | Input data has quality problems | Malformed text, corrupt image | Fix data quality issues |

### Error Examples

#### Invalid Parameter Example

```json
{
  "error": {
    "code": "invalid_parameter",
    "message": "Invalid parameters in request",
    "details": [
      {
        "field": "headline",
        "description": "Required field missing"
      },
      {
        "field": "platform",
        "description": "Must be one of: facebook, instagram, google, tiktok"
      }
    ]
  },
  "request_id": "req_7c63a1b2e5"
}
```

#### Model Error Example

```json
{
  "error": {
    "code": "model_error",
    "message": "Unable to generate prediction",
    "details": [
      {
        "field": "features",
        "description": "Insufficient context for reliable prediction"
      },
      {
        "field": "model",
        "description": "Model version 2.3.1 requires minimum confidence threshold not met"
      }
    ]
  },
  "request_id": "req_9d73a2b1c5"
}
```

#### Rate Limit Error Example

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "API rate limit exceeded",
    "details": [
      {
        "field": "rate_limit",
        "description": "Limit of 100 requests per minute exceeded. Reset in 45 seconds"
      }
    ]
  },
  "request_id": "req_8e73b2a1c6",
  "retry_after": 45
}
```

### Error Handling Best Practices

When integrating with the ML Inference API, follow these error handling best practices:

1. **Always check status codes** - Don't assume requests will succeed
2. **Implement exponential backoff** - For 429 and 5xx responses
3. **Log request IDs** - For troubleshooting and support
4. **Parse error details** - Handle specific error conditions appropriately
5. **Handle model-specific errors** - Different models may return specialized error details
6. **Provide fallback behavior** - Implement graceful degradation when the API is unavailable
7. **Monitor error rates** - Track API errors to detect problems early

### Validation Errors

For endpoints that perform validation of input data, the API returns detailed validation errors:

```json
{
  "error": {
    "code": "validation_failed",
    "message": "Validation failed",
    "details": [
      {
        "field": "date_range.start",
        "description": "Start date must be before end date"
      },
      {
        "field": "account_id",
        "description": "Account has insufficient historical data for prediction"
      }
    ]
  },
  "request_id": "req_7d63a2b1e4"
}
```

## Rate Limiting

The API implements rate limiting with these characteristics:

- **Limit Basis**: Per API key
- **Default Limit**: 100 requests per minute
- **Batch Size Limit**: 100 items per batch request
- **Concurrency Limit**: 10 concurrent requests per key

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1589547962
```

## Versioning

The API uses URL-based versioning:

- **Current Version**: v1
- **Deprecation Policy**: 6 months notice before version retirement
- **Version Lifecycle**: Alpha → Beta → Stable → Deprecated → Retired

Version information in requests:
```
https://api.within.co/v1/ml/ad-score
```

## Performance Characteristics

### Response Times

| Endpoint | P50 | P95 | P99 |
|----------|-----|-----|-----|
| /ad-score | 150ms | 300ms | 500ms |
| /account-health | 350ms | 700ms | 1200ms |
| /ad-sentiment | 80ms | 200ms | 350ms |
| /batch-predict | 500ms | 1200ms | 2000ms |

### Throughput

- **Single-instance Capacity**: 50 requests/second
- **Scaling Strategy**: Horizontal auto-scaling based on CPU utilization
- **Batch Processing Efficiency**: 5-10x more efficient than individual requests

### Caching

- **Cache Duration**: 1 hour for identical requests
- **Cache Key**: Hash of normalized request parameters
- **Cache Headers**: ETag and Last-Modified supported

## Integration Examples

### Python Example

```python
import requests

API_URL = "https://api.within.co/v1/ml"
API_KEY = "your_api_key"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Ad score prediction
ad_data = {
    "headline": "Limited Time Offer: 20% Off All Products",
    "description": "Shop our entire collection and save with this exclusive discount.",
    "cta": "Shop Now",
    "platform": "facebook",
    "industry": "retail"
}

response = requests.post(
    f"{API_URL}/ad-score",
    headers=headers,
    json=ad_data
)

if response.status_code == 200:
    result = response.json()
    print(f"Ad Score: {result['score']}")
    print(f"Confidence: {result['confidence']}")
    
    # Print top explanations
    for explanation in result['explanations'][:3]:
        print(f"- {explanation['description']} (Impact: {explanation['impact']})")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

### JavaScript Example

```javascript
const axios = require('axios');

const API_URL = 'https://api.within.co/v1/ml';
const API_KEY = 'your_api_key';

const headers = {
  Authorization: `Bearer ${API_KEY}`,
  'Content-Type': 'application/json'
};

// Account health assessment
const accountData = {
  account_id: 'act_123456',
  platform: 'facebook',
  time_range: 'last_30_days',
  include_recommendations: true
};

axios.post(`${API_URL}/account-health`, accountData, { headers })
  .then(response => {
    const result = response.data;
    console.log(`Health Score: ${result.health_score}`);
    console.log(`Trend: ${result.trend}`);
    console.log(`Risk Level: ${result.risk_level}`);
    
    // Print top issues
    console.log('Top Issues:');
    result.issues.slice(0, 3).forEach(issue => {
      console.log(`- ${issue.description} (Impact: ${issue.impact})`);
    });
    
    // Print top recommendations
    console.log('Top Recommendations:');
    result.recommendations.slice(0, 3).forEach(rec => {
      console.log(`- ${rec.description} (Potential Impact: ${rec.potential_impact})`);
    });
  })
  .catch(error => {
    console.error('Error:', error.response ? error.response.status : error.message);
    console.error(error.response ? error.response.data : error);
  });
```

## Monitoring and Debugging

### Request IDs

Every response includes a unique `request_id`