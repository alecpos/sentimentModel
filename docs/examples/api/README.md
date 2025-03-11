# API Examples

**IMPLEMENTATION STATUS: PARTIALLY_IMPLEMENTED**


This directory contains examples for working with the WITHIN API across different programming languages and scenarios.

## Available Examples

- `authentication.py` - Examples of different authentication methods
- `ad_score_prediction.py` - Predict ad effectiveness using the API
- `account_health.py` - Assess account health status
- `sentiment_analysis.py` - Analyze sentiment in ad text
- `batch_processing.py` - Process multiple ads in a single request
- `error_handling.py` - Properly handle API errors and exceptions
- `rate_limiting.py` - Implement backoff strategies for rate limits
- `webhook_setup.py` - Configure and use webhooks

## Basic Usage Example

```python
# Example of predicting ad score using Python
import requests
import json
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
    
    return {
        "X-Within-Access-Key": access_key,
        "X-Within-Timestamp": timestamp,
        "X-Within-Signature": base64.b64encode(signature).decode(),
        "Content-Type": "application/json"
    }

# Your API credentials
access_key = "your_access_key"
secret_key = "your_secret_key"

# Ad content to evaluate
ad_data = {
    "ad_content": "Limited time offer! 50% off our premium subscription plan. Sign up today!",
    "platform": "facebook",
    "industry": "technology",
    "campaign_objective": "conversions"
}

# Generate authentication headers
url = "https://api.within.co/api/v1/predict/ad_score"
headers = generate_auth_headers(access_key, secret_key, "POST", "/api/v1/predict/ad_score")

# Make API request
response = requests.post(url, headers=headers, data=json.dumps(ad_data))

# Process response
if response.status_code == 200:
    result = response.json()
    print(f"Ad Score: {result['score']}")
    print(f"Confidence: {result['confidence']}")
    print("Dimension Scores:")
    for dimension, score in result['dimension_scores'].items():
        print(f"  {dimension}: {score}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

## Language Support

Examples are available in:
- Python
- JavaScript
- Java
- Ruby
- PHP

## Requirements

Each example may have specific requirements. For Python examples, these are the common requirements:

```
requests>=2.28.1
python-dotenv>=0.21.0
```

## Using the Examples

1. Clone the repository
2. Navigate to the example directory
3. Install required dependencies
4. Update the example with your API credentials
5. Run the example

## See Also

- [API Reference Documentation](/docs/api/overview.md)
- [API Authentication Guide](/docs/api/authentication.md)
- [Error Handling Best Practices](/docs/api/error_codes.md) 