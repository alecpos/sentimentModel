# Quick Start Guide

**IMPLEMENTATION STATUS: IMPLEMENTED**


This guide will help you get started with the WITHIN Ad Score & Account Health Predictor system quickly. We'll cover basic setup, authentication, and walk through common use cases to help you start leveraging the system effectively.

## Prerequisites

Before starting, ensure you have:

- A WITHIN account (contact sales@within.co to create one if needed)
- API credentials (access key and secret key)
- Python 3.9+ installed (if using the Python SDK)
- Basic familiarity with REST APIs or Python

## Installation Options

### Python SDK (Recommended)

Install the Python SDK for the easiest integration:

```bash
pip install within-client
```

### API Direct Access

If you prefer direct API access, you can use any HTTP client. Examples in this guide will include both Python SDK and direct API calls using `curl` and Python's `requests` library.

## Authentication

### Using the Python SDK

```python
from within import Client

# Initialize client with your credentials
client = Client(
    access_key="your_access_key",
    secret_key="your_secret_key"
)

# Test authentication
account_info = client.get_account_info()
print(f"Authenticated as: {account_info['company_name']}")
```

### Using Direct API Calls

```python
import requests
import time
import hmac
import hashlib
import base64

def generate_auth_headers(access_key, secret_key, method, path):
    timestamp = str(int(time.time()))
    message = f"{method}\n{path}\n{timestamp}"
    signature = hmac.new(
        secret_key.encode(),
        message.encode(),
        hashlib.sha256
    ).digest()
    encoded_signature = base64.b64encode(signature).decode()
    
    return {
        "X-Within-Access-Key": access_key,
        "X-Within-Timestamp": timestamp,
        "X-Within-Signature": encoded_signature
    }

# Example API call
access_key = "your_access_key"
secret_key = "your_secret_key"
method = "GET"
path = "/api/v1/account"

headers = generate_auth_headers(access_key, secret_key, method, path)
response = requests.get(
    "https://api.within.co/api/v1/account",
    headers=headers
)

print(response.json())
```

## Common Use Cases

### 1. Predict Ad Effectiveness

Evaluate the effectiveness of ad content before launching campaigns.

#### Using Python SDK

```python
# Create ad data
ad_data = {
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

# Generate prediction
prediction = client.predict_ad_score(ad_data)

# Print results
print(f"Ad Score: {prediction['score']}/100")
print(f"Confidence: {prediction['confidence']}")
print("\nComponent Scores:")
for component, score in prediction['score_components'].items():
    print(f"  {component}: {score}")
```

#### Using Direct API

```bash
curl -X POST "https://api.within.co/api/v1/ad-score/predict" \
  -H "Content-Type: application/json" \
  -H "X-Within-Access-Key: your_access_key" \
  -H "X-Within-Timestamp: timestamp" \
  -H "X-Within-Signature: signature" \
  -d '{
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
  }'
```

### 2. Assess Account Health

Monitor the health of your advertising accounts and identify optimization opportunities.

#### Using Python SDK

```python
# Get account health score
account_health = client.get_account_health(
    account_id="123456789",
    platform="google"
)

# Print results
print(f"Account Health Score: {account_health['health_score']}/100")
print(f"Status: {account_health['health_category']}")

# Print risk factors
print("\nRisk Factors:")
for risk in account_health['risk_factors']:
    print(f"  • {risk['description']} (Impact: {risk['impact']}/10)")

# Print recommendations
print("\nRecommendations:")
for rec in account_health['recommendations']:
    print(f"  • {rec['description']} (Est. Impact: {rec['estimated_impact']})")
```

#### Using Direct API

```bash
curl -X GET "https://api.within.co/api/v1/account-health/score?account_id=123456789&platform=google" \
  -H "X-Within-Access-Key: your_access_key" \
  -H "X-Within-Timestamp: timestamp" \
  -H "X-Within-Signature: signature"
```

### 3. Analyze Ad Text Sentiment

Analyze the sentiment and emotional aspects of your ad copy.

#### Using Python SDK

```python
# Analyze ad text sentiment
sentiment = client.analyze_sentiment(
    text="Limited time offer! Get 50% off our premium package and transform your results today.",
    include_aspects=True
)

# Print results
print(f"Sentiment: {sentiment['sentiment']}")
print(f"Confidence: {sentiment['confidence']}")
print(f"Intensity: {sentiment['intensity']}")

# Print emotional aspects
print("\nEmotional Aspects:")
for aspect, score in sentiment['aspects'].items():
    print(f"  {aspect}: {score}")
```

#### Using Direct API

```bash
curl -X POST "https://api.within.co/api/v1/nlp/sentiment" \
  -H "Content-Type: application/json" \
  -H "X-Within-Access-Key: your_access_key" \
  -H "X-Within-Timestamp: timestamp" \
  -H "X-Within-Signature: signature" \
  -d '{
    "text": "Limited time offer! Get 50% off our premium package and transform your results today.",
    "include_aspects": true
  }'
```

### 4. Compare Multiple Ad Variations

Compare different ad variations to identify the most effective options.

#### Using Python SDK

```python
# Define multiple ad variations
ad_variations = [
    {
        "id": "variation_1",
        "headline": "Limited Time Offer: 20% Off All Products",
        "description": "Shop our entire collection and save with this exclusive discount.",
        "cta": "Shop Now"
    },
    {
        "id": "variation_2",
        "headline": "Save 20% Sitewide - This Week Only!",
        "description": "Exclusive discount on all products. Don't miss this opportunity.",
        "cta": "Get Discount"
    },
    {
        "id": "variation_3",
        "headline": "20% Discount On Everything - Limited Time",
        "description": "Use code SAVE20 at checkout to apply your discount on any product.",
        "cta": "Use Code SAVE20"
    }
]

# Common parameters for all variations
common_params = {
    "platform": "facebook",
    "target_audience": ["fashion_shoppers", "deal_seekers"]
}

# Predict scores for all variations
comparison = client.compare_ad_variations(ad_variations, **common_params)

# Print comparison results
print("Ad Variation Comparison:")
for result in comparison['results']:
    print(f"\nVariation: {result['id']}")
    print(f"  Score: {result['score']}/100")
    print(f"  Confidence: {result['confidence']}")
    print("  Top Strengths:")
    for strength in result['strengths'][:2]:
        print(f"    • {strength}")
```

### 5. Get Ad Performance Analytics

Access performance analytics for your ads across platforms.

#### Using Python SDK

```python
# Get performance analytics
analytics = client.get_ad_performance(
    ad_ids=["ad_123", "ad_456", "ad_789"],
    time_range="last_30_days",
    metrics=["impressions", "clicks", "conversions", "ctr", "conversion_rate", "cpa"],
    compare_to="previous_period"
)

# Print results
print("Ad Performance Analytics:")
for ad_id, data in analytics['ads'].items():
    print(f"\nAd: {ad_id}")
    for metric, value in data['metrics'].items():
        change = data['changes'][metric]
        change_symbol = "▲" if change > 0 else "▼" if change < 0 else "◆"
        print(f"  {metric}: {value} {change_symbol} {abs(change)}%")
```

#### Using Direct API

```bash
curl -X GET "https://api.within.co/api/v1/analytics/ad-performance?ad_ids=ad_123,ad_456,ad_789&time_range=last_30_days&metrics=impressions,clicks,conversions,ctr,conversion_rate,cpa&compare_to=previous_period" \
  -H "X-Within-Access-Key: your_access_key" \
  -H "X-Within-Timestamp: timestamp" \
  -H "X-Within-Signature: signature"
```

## Using the Dashboard

In addition to the API and SDK, WITHIN provides a web dashboard for easy access to all features.

1. Log in to the dashboard at [https://app.within.co](https://app.within.co)
2. Navigate between different sections using the left sidebar:
   - **Overview**: Summary of account performance and health
   - **Ad Score**: Ad effectiveness prediction and optimization
   - **Account Health**: Detailed account health monitoring
   - **Analytics**: Performance analytics and reporting
   - **Settings**: API credentials and user management

![Dashboard Overview](../images/dashboard_overview.png)

## Batch Processing

For bulk operations, use the batch processing endpoints:

### Using Python SDK

```python
# Batch ad score prediction
ad_data_list = [
    {"ad_id": "ad1", "headline": "Headline 1", "description": "Description 1", "cta": "CTA 1"},
    {"ad_id": "ad2", "headline": "Headline 2", "description": "Description 2", "cta": "CTA 2"},
    # ... more ads
]

common_params = {
    "platform": "facebook",
    "target_audience": ["general"]
}

batch_results = client.batch_predict_ad_scores(ad_data_list, **common_params)

# Process batch results
for result in batch_results['predictions']:
    print(f"Ad {result['ad_id']}: Score {result['score']}/100")
```

## Error Handling

Handle potential errors in your integration:

```python
from within.exceptions import WithinApiError, AuthenticationError, ValidationError

try:
    result = client.predict_ad_score(ad_data)
except ValidationError as e:
    print(f"Invalid input data: {e}")
    # Handle validation errors (e.g., fix input data)
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
    # Handle authentication errors (e.g., refresh credentials)
except WithinApiError as e:
    print(f"API error: {e.code} - {e.message}")
    # Handle other API errors
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle unexpected errors
```

## Rate Limits

Be aware of rate limits when making API calls:

| Endpoint Type | Rate Limit |
|---------------|------------|
| Prediction endpoints | 100 requests per minute |
| Batch prediction endpoints | 10 requests per minute |
| Analytics endpoints | 30 requests per minute |
| Account health endpoints | 30 requests per minute |

Rate limit headers are included in all responses. Monitor them to avoid hitting limits:

```python
response = requests.post(
    "https://api.within.co/api/v1/ad-score/predict",
    headers=headers,
    json=ad_data
)

# Check rate limit headers
rate_limit = int(response.headers.get("X-RateLimit-Limit", 0))
remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
reset_time = int(response.headers.get("X-RateLimit-Reset", 0))

print(f"Rate limit: {remaining}/{rate_limit} requests remaining")
print(f"Rate limit resets at: {datetime.fromtimestamp(reset_time)}")
```

## Next Steps

Now that you've completed the quick start guide, explore these resources to learn more:

- [API Reference](../api/overview.md): Complete API documentation
- [SDK Documentation](../api/python_sdk.md): Detailed Python SDK documentation
- [Model Documentation](../implementation/ml/index.md): Learn about the ML models
- [Dashboard Guide](../user_guides/dashboards.md): In-depth guide to the dashboard
- [Integration Examples](../examples/): Example code for common scenarios
- [Troubleshooting Guide](../maintenance/troubleshooting.md): Solutions to common issues

For additional help, contact our support team at support@within.co. 