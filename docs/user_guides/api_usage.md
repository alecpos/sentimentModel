# API Usage Guide

**IMPLEMENTATION STATUS: PARTIALLY_IMPLEMENTED**


This guide provides detailed instructions and examples for using the WITHIN API effectively. Whether you're integrating WITHIN's predictive capabilities into your applications or automating your analytics workflows, this guide will help you leverage the full power of the API.

## Table of Contents

- [Authentication](#authentication)
- [Making API Requests](#making-api-requests)
- [Response Formats](#response-formats)
- [Error Handling](#error-handling)
- [Ad Score Prediction](#ad-score-prediction)
- [Account Health Assessment](#account-health-assessment)
- [Sentiment Analysis](#sentiment-analysis)
- [Analytics & Reporting](#analytics--reporting)
- [Batch Processing](#batch-processing)
- [Rate Limiting](#rate-limiting)
- [Best Practices](#best-practices)
- [Client Libraries](#client-libraries)
- [Webhooks](#webhooks)

## Authentication

All API requests must be authenticated using API keys. Each request includes authentication information in the HTTP headers.

### Obtaining API Keys

1. Log in to the [WITHIN Dashboard](https://app.within.co)
2. Navigate to Settings > API Keys
3. Click "Create New API Key"
4. Save both the Access Key and Secret Key securely

### Authentication Headers

Every API request must include the following headers:

- `X-Within-Access-Key`: Your access key
- `X-Within-Timestamp`: Current Unix timestamp (seconds since epoch)
- `X-Within-Signature`: HMAC-SHA256 signature of the request

### Generating the Signature

The signature is a HMAC-SHA256 hash of a canonical request string, using your Secret Key. The canonical request string includes:

1. HTTP method (GET, POST, etc.)
2. Request path (e.g., `/api/v1/ad-score/predict`)
3. Timestamp

Example in Python:

```python
import time
import hmac
import hashlib
import base64

def generate_signature(secret_key, method, path, timestamp):
    """Generate HMAC signature for API authentication."""
    message = f"{method}\n{path}\n{timestamp}"
    signature = hmac.new(
        secret_key.encode(),
        message.encode(),
        hashlib.sha256
    ).digest()
    return base64.b64encode(signature).decode()

def generate_auth_headers(access_key, secret_key, method, path):
    """Generate all required authentication headers."""
    timestamp = str(int(time.time()))
    signature = generate_signature(secret_key, method, path, timestamp)
    
    return {
        "X-Within-Access-Key": access_key,
        "X-Within-Timestamp": timestamp,
        "X-Within-Signature": signature
    }
```

## Making API Requests

### API Base URL

All API requests should be made to the following base URL:

```
https://api.within.co/api/v1
```

### Request Content Types

- For `POST` and `PUT` requests, set `Content-Type: application/json`
- For file uploads, use `Content-Type: multipart/form-data`

### Basic Request Example

```python
import requests

# Your API credentials
access_key = "your_access_key"
secret_key = "your_secret_key"

# Endpoint and method
method = "GET"
path = "/api/v1/account"
url = f"https://api.within.co{path}"

# Generate authentication headers
headers = generate_auth_headers(access_key, secret_key, method, path)

# Make the request
response = requests.get(url, headers=headers)

# Check response
if response.status_code == 200:
    account_info = response.json()
    print(f"Account name: {account_info['data']['company_name']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

## Response Formats

All API responses follow a consistent format:

### Success Response

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

### Error Response

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

## Error Handling

### Common Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | VALIDATION_ERROR | Invalid input data |
| 401 | UNAUTHORIZED | Missing or invalid credentials |
| 403 | FORBIDDEN | Insufficient permissions |
| 404 | NOT_FOUND | Resource not found |
| 409 | CONFLICT | Resource conflict |
| 429 | RATE_LIMITED | Too many requests |
| 500 | SERVER_ERROR | Internal server error |

### Handling Errors in Your Application

```python
response = requests.post(url, headers=headers, json=payload)

if response.status_code >= 400:
    error_data = response.json()
    error_code = error_data.get("error", {}).get("code", "UNKNOWN_ERROR")
    error_message = error_data.get("error", {}).get("message", "Unknown error")
    
    if error_code == "VALIDATION_ERROR":
        # Handle validation errors
        details = error_data.get("error", {}).get("details", [])
        for detail in details:
            print(f"Validation error in {detail.get('field')}: {detail.get('issue')}")
        
        # Fix the validation issues and retry
        
    elif error_code == "UNAUTHORIZED":
        # Handle authentication errors
        print("Authentication failed. Check your API credentials.")
        
        # Refresh credentials or prompt for new ones
        
    elif error_code == "RATE_LIMITED":
        # Handle rate limiting
        retry_after = int(response.headers.get("Retry-After", 60))
        print(f"Rate limit exceeded. Retry after {retry_after} seconds.")
        
        # Implement exponential backoff or pause processing
        
    else:
        # Handle other errors
        print(f"API error: {error_code} - {error_message}")
```

## Ad Score Prediction

The Ad Score Prediction API allows you to predict the effectiveness of ad content before launching campaigns.

### Single Ad Prediction

**Endpoint**: `POST /api/v1/ad-score/predict`

**Request**:

```python
import requests

# Your API credentials
access_key = "your_access_key"
secret_key = "your_secret_key"

# Endpoint and method
method = "POST"
path = "/api/v1/ad-score/predict"
url = f"https://api.within.co{path}"

# Generate authentication headers
headers = generate_auth_headers(access_key, secret_key, method, path)
headers["Content-Type"] = "application/json"

# Request payload
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

# Make the request
response = requests.post(url, headers=headers, json=payload)

# Process the response
if response.status_code == 200:
    result = response.json()
    prediction = result["data"]
    
    print(f"Ad Score: {prediction['score']}/100")
    print(f"Confidence: {prediction['confidence']}")
    print("\nComponent Scores:")
    for component, score in prediction['score_components'].items():
        print(f"  {component}: {score}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

**Response**:

```json
{
  "data": {
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
  },
  "meta": {
    "request_id": "req_abc123",
    "timestamp": "2023-04-15T10:30:45Z",
    "processing_time_ms": 235
  }
}
```

### Getting Prediction Explanation

**Endpoint**: `GET /api/v1/ad-score/explain/{prediction_id}`

```python
# Endpoint and method
method = "GET"
prediction_id = "ad_score_123456"
path = f"/api/v1/ad-score/explain/{prediction_id}"
url = f"https://api.within.co{path}"

# Generate authentication headers
headers = generate_auth_headers(access_key, secret_key, method, path)

# Make the request
response = requests.get(url, headers=headers)

# Process the explanation
if response.status_code == 200:
    result = response.json()
    explanation = result["data"]
    
    print("Score Explanation:")
    print(f"Top strengths:")
    for strength in explanation["strengths"]:
        print(f"  • {strength}")
        
    print(f"\nAreas for improvement:")
    for improvement in explanation["improvements"]:
        print(f"  • {improvement}")
        
    print(f"\nFeature importance:")
    for feature, importance in explanation["feature_importance"].items():
        print(f"  {feature}: {importance}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

## Account Health Assessment

The Account Health API allows you to evaluate the health of your advertising accounts and identify optimization opportunities.

### Get Account Health Score

**Endpoint**: `GET /api/v1/account-health/score`

```python
# Endpoint and method
method = "GET"
path = "/api/v1/account-health/score"
url = f"https://api.within.co{path}"

# Query parameters
params = {
    "account_id": "123456789",
    "platform": "google"
}

# Generate authentication headers
headers = generate_auth_headers(access_key, secret_key, method, path)

# Make the request
response = requests.get(url, headers=headers, params=params)

# Process the response
if response.status_code == 200:
    result = response.json()
    health = result["data"]
    
    print(f"Account Health Score: {health['health_score']}/100")
    print(f"Health Category: {health['health_category']}")
    
    print("\nRisk Factors:")
    for risk in health["risk_factors"]:
        print(f"  • {risk['description']} (Impact: {risk['impact']}/10)")
        
    print("\nRecommendations:")
    for rec in health["recommendations"]:
        print(f"  • {rec['description']}")
        print(f"    Estimated Impact: {rec['estimated_impact']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### Get Account Health Trends

**Endpoint**: `GET /api/v1/account-health/trends`

```python
# Endpoint and method
method = "GET"
path = "/api/v1/account-health/trends"
url = f"https://api.within.co{path}"

# Query parameters
params = {
    "account_id": "123456789",
    "platform": "google",
    "time_range": "last_90_days",
    "granularity": "weekly"
}

# Generate authentication headers
headers = generate_auth_headers(access_key, secret_key, method, path)

# Make the request
response = requests.get(url, headers=headers, params=params)

# Process the response
if response.status_code == 200:
    result = response.json()
    trends = result["data"]
    
    print("Account Health Trends:")
    for point in trends["trends"]:
        print(f"  {point['date']}: {point['health_score']}/100 ({point['health_category']})")
        
    print("\nMetric Trends:")
    for metric, values in trends["metrics"].items():
        print(f"  {metric}:")
        for point in values:
            print(f"    {point['date']}: {point['value']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

## Sentiment Analysis

The Sentiment Analysis API allows you to analyze the sentiment and emotional aspects of ad text.

### Analyze Text Sentiment

**Endpoint**: `POST /api/v1/nlp/sentiment`

```python
# Endpoint and method
method = "POST"
path = "/api/v1/nlp/sentiment"
url = f"https://api.within.co{path}"

# Generate authentication headers
headers = generate_auth_headers(access_key, secret_key, method, path)
headers["Content-Type"] = "application/json"

# Request payload
payload = {
    "text": "Limited time offer! Get 50% off our premium package and transform your results today.",
    "include_aspects": True
}

# Make the request
response = requests.post(url, headers=headers, json=payload)

# Process the response
if response.status_code == 200:
    result = response.json()
    sentiment = result["data"]
    
    print(f"Sentiment: {sentiment['sentiment']}")
    print(f"Confidence: {sentiment['confidence']}")
    print(f"Intensity: {sentiment['intensity']}")
    
    print("\nEmotional Aspects:")
    for aspect, score in sentiment["aspects"].items():
        print(f"  {aspect}: {score}")
        
    if "token_scores" in sentiment:
        print("\nToken-level Sentiment:")
        for token in sentiment["token_scores"]:
            print(f"  {token['token']}: {token['score']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### Batch Sentiment Analysis

**Endpoint**: `POST /api/v1/nlp/sentiment/batch`

```python
# Endpoint and method
method = "POST"
path = "/api/v1/nlp/sentiment/batch"
url = f"https://api.within.co{path}"

# Generate authentication headers
headers = generate_auth_headers(access_key, secret_key, method, path)
headers["Content-Type"] = "application/json"

# Request payload
payload = {
    "texts": [
        "Limited time offer! Get 50% off our premium package.",
        "Discover the most reliable solution for professionals.",
        "Stop wasting money on ineffective products. Try our solution."
    ],
    "include_aspects": True
}

# Make the request
response = requests.post(url, headers=headers, json=payload)

# Process the response
if response.status_code == 200:
    result = response.json()
    sentiments = result["data"]["results"]
    
    for i, sentiment in enumerate(sentiments):
        print(f"\nText {i+1}:")
        print(f"  Sentiment: {sentiment['sentiment']}")
        print(f"  Confidence: {sentiment['confidence']}")
        print(f"  Aspects: {', '.join([f'{k}:{v:.2f}' for k, v in sentiment['aspects'].items()])}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

## Analytics & Reporting

The Analytics API allows you to access performance data and generate reports.

### Get Ad Performance Analytics

**Endpoint**: `GET /api/v1/analytics/ad-performance`

```python
# Endpoint and method
method = "GET"
path = "/api/v1/analytics/ad-performance"
url = f"https://api.within.co{path}"

# Query parameters
params = {
    "ad_ids": "ad_123,ad_456,ad_789",
    "time_range": "last_30_days",
    "metrics": "impressions,clicks,conversions,ctr,conversion_rate,cpa",
    "compare_to": "previous_period"
}

# Generate authentication headers
headers = generate_auth_headers(access_key, secret_key, method, path)

# Make the request
response = requests.get(url, headers=headers, params=params)

# Process the response
if response.status_code == 200:
    result = response.json()
    analytics = result["data"]
    
    print("Ad Performance Analytics:")
    for ad_id, data in analytics["ads"].items():
        print(f"\nAd: {ad_id}")
        for metric, value in data["metrics"].items():
            change = data["changes"][metric]
            change_symbol = "▲" if change > 0 else "▼" if change < 0 else "◆"
            print(f"  {metric}: {value} {change_symbol} {abs(change)}%")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### Get Platform Comparison

**Endpoint**: `GET /api/v1/analytics/platform-comparison`

```python
# Endpoint and method
method = "GET"
path = "/api/v1/analytics/platform-comparison"
url = f"https://api.within.co{path}"

# Query parameters
params = {
    "time_range": "last_30_days",
    "metrics": "impressions,clicks,conversions,ctr,conversion_rate,cpa,roas",
    "platforms": "facebook,google,tiktok"
}

# Generate authentication headers
headers = generate_auth_headers(access_key, secret_key, method, path)

# Make the request
response = requests.get(url, headers=headers, params=params)

# Process the response
if response.status_code == 200:
    result = response.json()
    comparison = result["data"]
    
    print("Platform Comparison:")
    for platform, metrics in comparison["platforms"].items():
        print(f"\n{platform.capitalize()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
        
    print("\nPerformance Index (Relative to Average):")
    for platform, index in comparison["performance_index"].items():
        print(f"  {platform.capitalize()}: {index}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

## Batch Processing

For efficiency when processing multiple items, use the batch endpoints.

### Batch Ad Score Prediction

**Endpoint**: `POST /api/v1/ad-score/batch-predict`

```python
# Endpoint and method
method = "POST"
path = "/api/v1/ad-score/batch-predict"
url = f"https://api.within.co{path}"

# Generate authentication headers
headers = generate_auth_headers(access_key, secret_key, method, path)
headers["Content-Type"] = "application/json"

# Request payload
payload = {
    "ads": [
        {
            "id": "ad1",
            "ad_content": {
                "headline": "Limited Time Offer: 20% Off All Products",
                "description": "Shop our entire collection and save with this exclusive discount.",
                "cta": "Shop Now"
            }
        },
        {
            "id": "ad2",
            "ad_content": {
                "headline": "New Collection Now Available",
                "description": "Check out our latest styles for the new season.",
                "cta": "Browse Collection"
            }
        },
        {
            "id": "ad3",
            "ad_content": {
                "headline": "Free Shipping on Orders Over $50",
                "description": "Limited time offer for all customers.",
                "cta": "Start Shopping"
            }
        }
    ],
    "common_params": {
        "platform": "facebook",
        "target_audience": ["fashion_shoppers", "general"]
    }
}

# Make the request
response = requests.post(url, headers=headers, json=payload)

# Process the response
if response.status_code == 200:
    result = response.json()
    predictions = result["data"]["predictions"]
    
    print("Batch Prediction Results:")
    for prediction in predictions:
        print(f"\nAd ID: {prediction['id']}")
        print(f"  Score: {prediction['score']}/100")
        print(f"  Confidence: {prediction['confidence']}")
        for component, score in prediction['score_components'].items():
            print(f"  {component}: {score}")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

## Rate Limiting

The API implements rate limiting to ensure fair usage. Different endpoints have different rate limits:

| Endpoint Type | Rate Limit |
|---------------|------------|
| Prediction endpoints | 100 requests per minute |
| Batch prediction endpoints | 10 requests per minute |
| Analytics endpoints | 30 requests per minute |
| Account health endpoints | 30 requests per minute |

### Rate Limit Headers

Rate limit information is included in response headers:

- `X-RateLimit-Limit`: Maximum requests per minute
- `X-RateLimit-Remaining`: Remaining requests in the current window
- `X-RateLimit-Reset`: Unix timestamp when the rate limit resets

### Handling Rate Limits

```python
response = requests.post(url, headers=headers, json=payload)

# Check rate limit headers
rate_limit = int(response.headers.get("X-RateLimit-Limit", 0))
remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
reset_time = int(response.headers.get("X-RateLimit-Reset", 0))

print(f"Rate limit: {remaining}/{rate_limit} requests remaining")
print(f"Rate limit resets at: {datetime.fromtimestamp(reset_time)}")

# If rate limit is close to being reached, slow down
if remaining < rate_limit * 0.1:  # Less than 10% remaining
    wait_time = max(1, reset_time - time.time())
    print(f"Rate limit almost reached. Waiting {wait_time:.0f} seconds.")
    time.sleep(wait_time)

# If rate limit is exceeded (status code 429)
if response.status_code == 429:
    retry_after = int(response.headers.get("Retry-After", 60))
    print(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
    time.sleep(retry_after)
    
    # Retry the request
    response = requests.post(url, headers=headers, json=payload)
```

## Best Practices

### Efficiency

1. **Use Batch Processing**: When processing multiple items, use batch endpoints instead of making individual requests
2. **Cache Results**: Cache prediction results for identical inputs to reduce API calls
3. **Optimize Payload Size**: Include only necessary fields in your requests

### Reliability

1. **Implement Retry Logic**: Retry failed requests with exponential backoff
2. **Handle Rate Limits**: Monitor rate limit headers and adjust request rates accordingly
3. **Validate Inputs**: Validate inputs before sending to avoid validation errors

### Security

1. **Secure API Keys**: Store your API keys securely and never expose them in client-side code
2. **Rotate Keys Regularly**: Rotate your API keys periodically for security
3. **Use HTTPS**: Always use HTTPS for API requests

### Example Retry Logic

```python
import time
import random

def make_api_request_with_retry(method, url, headers, json=None, params=None, max_retries=3, base_delay=1):
    """Make an API request with exponential backoff retry logic."""
    retries = 0
    
    while retries <= max_retries:
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=json)
            # Add other methods as needed
            
            # If rate limited, retry after the specified time
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                print(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
                continue
                
            # If successful or non-retriable error, return response
            if response.status_code < 500:
                return response
                
            # For server errors, retry with backoff
            print(f"Server error {response.status_code}. Retrying...")
            
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"Connection error: {e}. Retrying...")
        
        # Exponential backoff with jitter
        delay = base_delay * (2 ** retries) + random.uniform(0, 0.5)
        time.sleep(delay)
        retries += 1
    
    # If all retries failed, raise exception
    raise Exception(f"Failed after {max_retries} retries")
```

## Client Libraries

WITHIN provides official client libraries for several programming languages:

### Python SDK

```bash
pip install within-client
```

```python
from within import Client

# Initialize client
client = Client(
    access_key="your_access_key",
    secret_key="your_secret_key"
)

# Predict ad score
prediction = client.predict_ad_score({
    "ad_content": {
        "headline": "Limited Time Offer: 20% Off All Products",
        "description": "Shop our entire collection and save with this exclusive discount.",
        "cta": "Shop Now"
    },
    "platform": "facebook",
    "target_audience": ["fashion_shoppers", "deal_seekers"]
})

print(f"Ad Score: {prediction['score']}/100")
```

### JavaScript SDK

```bash
npm install within-js-client
```

```javascript
const Within = require('within-js-client');

// Initialize client
const client = new Within.Client({
    accessKey: 'your_access_key',
    secretKey: 'your_secret_key'
});

// Predict ad score
client.predictAdScore({
    adContent: {
        headline: 'Limited Time Offer: 20% Off All Products',
        description: 'Shop our entire collection and save with this exclusive discount.',
        cta: 'Shop Now'
    },
    platform: 'facebook',
    targetAudience: ['fashion_shoppers', 'deal_seekers']
})
.then(prediction => {
    console.log(`Ad Score: ${prediction.score}/100`);
})
.catch(error => {
    console.error('Error:', error);
});
```

## Webhooks

WITHIN provides webhooks for event-based notifications:

### Available Webhook Events

- `prediction.completed`: Triggered when a prediction is completed
- `batch_prediction.completed`: Triggered when a batch prediction is completed
- `account_health.updated`: Triggered when account health is updated
- `model.updated`: Triggered when a model is updated

### Configuring Webhooks

1. Log in to the [WITHIN Dashboard](https://app.within.co)
2. Navigate to Settings > Webhooks
3. Click "Add Webhook"
4. Enter the webhook URL
5. Select the events to subscribe to
6. Optionally, add a secret for verifying webhook signatures

### Webhook Payload Example

```json
{
  "event": "prediction.completed",
  "timestamp": "2023-04-15T10:30:45Z",
  "data": {
    "prediction_id": "ad_score_123456",
    "score": 78.5,
    "confidence": 0.92,
    "model_version": "2.1.0"
  }
}
```

### Verifying Webhook Signatures

Webhook requests include a signature in the `X-Within-Signature` header. Verify this signature to ensure the webhook came from WITHIN:

```python
import hmac
import hashlib

def verify_webhook_signature(payload, signature, secret):
    """Verify the webhook signature."""
    computed_signature = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(computed_signature, signature)

# In your webhook handler
def webhook_handler(request):
    payload = request.body
    signature = request.headers.get("X-Within-Signature")
    secret = "your_webhook_secret"
    
    if verify_webhook_signature(payload, signature, secret):
        # Process the webhook
        event_data = json.loads(payload)
        event_type = event_data["event"]
        
        if event_type == "prediction.completed":
            prediction = event_data["data"]
            print(f"Prediction completed: {prediction['prediction_id']}")
            print(f"Score: {prediction['score']}/100")
            
        # Handle other event types
        
        return {"status": "success"}
    else:
        # Invalid signature
        return {"status": "error", "message": "Invalid signature"}, 401
```

## Additional Resources

- [API Reference](../api/overview.md): Complete API documentation
- [Authentication Guide](../api/authentication.md): Detailed authentication instructions
- [Error Codes Reference](../api/error_codes.md): Full list of error codes and handling
- [Webhooks Guide](../api/webhooks.md): Detailed webhooks documentation
- [SDK Documentation](../api/python_sdk.md): Complete SDK documentation
- [Rate Limits Guide](../api/rate_limits.md): Detailed rate limits information 