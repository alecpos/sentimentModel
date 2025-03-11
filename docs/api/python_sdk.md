# WITHIN Python SDK

**IMPLEMENTATION STATUS: PARTIALLY_IMPLEMENTED**


This document provides comprehensive guidance on using the WITHIN Python SDK to interact with the Ad Score Predictor, Account Health Predictor, and Ad Sentiment Analyzer models.

## Table of Contents

1. [Installation](#installation)
2. [Authentication](#authentication)
3. [Basic Usage](#basic-usage)
4. [Ad Score Predictor API](#ad-score-predictor-api)
5. [Account Health Predictor API](#account-health-predictor-api)
6. [Ad Sentiment Analyzer API](#ad-sentiment-analyzer-api)
7. [Batch Processing](#batch-processing)
8. [Error Handling](#error-handling)
9. [Rate Limiting](#rate-limiting)
10. [Advanced Configuration](#advanced-configuration)
11. [Logging and Debugging](#logging-and-debugging)
12. [Examples](#examples)
13. [Changelog](#changelog)

## Installation

Install the WITHIN Python SDK using pip:

```bash
pip install within-sdk
```

To install a specific version:

```bash
pip install within-sdk==2.1.0
```

Requirements:
- Python 3.9+
- Dependencies:
  - requests>=2.25.0
  - pandas>=1.2.0
  - numpy>=1.20.0
  - PyJWT>=2.0.0
  - tenacity>=7.0.0

## Authentication

The SDK supports two authentication methods: API key and OAuth.

### API Key Authentication

```python
from within import Client

# Initialize with API key
client = Client(api_key="your_api_key")
```

### OAuth Authentication

```python
from within import Client

# Initialize with OAuth credentials
client = Client(
    client_id="your_client_id",
    client_secret="your_client_secret",
    oauth_scope="prediction:read prediction:write"
)
```

### Setting Default Timeout and Retries

```python
from within import Client

# Initialize with custom timeout and retry settings
client = Client(
    api_key="your_api_key",
    timeout=60,  # seconds
    max_retries=3,
    retry_backoff_factor=0.5
)
```

## Basic Usage

The SDK provides access to all WITHIN prediction models through a unified client interface.

```python
from within import Client

# Initialize client
client = Client(api_key="your_api_key")

# Check service status
status = client.get_status()
print(f"API Status: {status['status']}")
print(f"Available Models: {', '.join(status['available_models'])}")

# Get model versions
versions = client.get_model_versions()
for model, version in versions.items():
    print(f"{model}: v{version}")
```

## Ad Score Predictor API

The Ad Score Predictor evaluates the quality and potential performance of ad content.

### Predict Ad Score

```python
# Basic ad score prediction
ad_score = client.ad_score.predict(
    ad_content="Experience the ultimate comfort with our premium mattress.",
    platform="facebook",
    ad_format="image",
    industry="home_goods"
)

print(f"Overall Score: {ad_score['overall_score']}")
print(f"Engagement Probability: {ad_score['engagement_probability']:.2f}")
print(f"Conversion Probability: {ad_score['conversion_probability']:.2f}")
```

### Detailed Ad Analysis

```python
# Get detailed analysis with component scores
analysis = client.ad_score.analyze(
    ad_content="Experience the ultimate comfort with our premium mattress.",
    platform="facebook",
    ad_format="image",
    industry="home_goods",
    include_components=True,
    include_explanations=True
)

print(f"Overall Score: {analysis['overall_score']}")

# Component scores
for component, score in analysis['component_scores'].items():
    print(f"{component}: {score:.2f}")

# Explanations
for explanation in analysis['explanations']:
    print(f"- {explanation}")
```

### Ad Comparison

```python
# Compare multiple ad variants
variants = [
    "Experience the ultimate comfort with our premium mattress.",
    "Sleep better tonight with the most comfortable mattress ever made.",
    "Our mattress: Scientifically designed for the perfect night's sleep."
]

comparison = client.ad_score.compare(
    ad_contents=variants,
    platform="facebook",
    ad_format="image",
    industry="home_goods"
)

for i, result in enumerate(comparison['results']):
    print(f"Variant {i+1}: Score {result['overall_score']:.2f}")

print(f"Best Variant: {comparison['best_variant_index'] + 1}")
print(f"Improvement over baseline: {comparison['improvement_percentage']:.1f}%")
```

### Ad Optimization Suggestions

```python
# Get optimization suggestions
suggestions = client.ad_score.get_suggestions(
    ad_content="Experience the ultimate comfort with our premium mattress.",
    platform="facebook",
    ad_format="image",
    industry="home_goods",
    target_score=85
)

print("Optimization Suggestions:")
for suggestion in suggestions['suggestions']:
    print(f"- {suggestion['description']}")
    print(f"  Estimated impact: +{suggestion['estimated_impact']:.1f} points")
```

## Account Health Predictor API

The Account Health Predictor evaluates advertising account performance and provides recommendations.

### Predict Account Health

```python
import pandas as pd

# Prepare account data
metrics_data = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=30, freq='D'),
    'impressions': [...],  # List of daily impressions
    'clicks': [...],       # List of daily clicks
    'conversions': [...],  # List of daily conversions
    'spend': [...]         # List of daily spend
})

# Basic account health prediction
health = client.account_health.predict(
    account_id="123456789",
    platform="facebook",
    metrics_data=metrics_data,
    include_recommendations=True
)

print(f"Account Health Score: {health['health_score']}")
print(f"Risk Level: {health['risk_level']}")
print(f"30-Day Trend: {health['trend_direction']}")

# Recommendations
for rec in health['recommendations']:
    print(f"- {rec['description']}")
    print(f"  Priority: {rec['priority']}")
    print(f"  Estimated Impact: {rec['estimated_impact']}")
```

### Detailed Account Analysis

```python
# Get detailed account analysis
analysis = client.account_health.analyze(
    account_id="123456789",
    platform="facebook",
    metrics_data=metrics_data,
    account_structure=account_structure_dict,  # Account campaign/ad set structure
    include_component_scores=True,
    include_forecasts=True,
    include_anomalies=True
)

print(f"Account Health Score: {analysis['health_score']}")

# Component health scores
for component, score in analysis['component_scores'].items():
    print(f"{component}: {score:.2f}")

# Performance forecasts
for metric, forecast in analysis['forecasts'].items():
    print(f"{metric} 30-day forecast: {forecast['value']:.2f}")
    print(f"Forecast confidence: {forecast['confidence']:.2f}")

# Detected anomalies
for anomaly in analysis['anomalies']:
    print(f"Anomaly in {anomaly['metric']}: {anomaly['description']}")
    print(f"Detected on: {anomaly['date']}")
    print(f"Severity: {anomaly['severity']}")
```

### Account Optimization Plan

```python
# Generate comprehensive optimization plan
plan = client.account_health.generate_plan(
    account_id="123456789",
    platform="facebook",
    metrics_data=metrics_data,
    target_health_score=85,
    target_metrics={"cpa": 15.0, "roas": 4.0}
)

print(f"Current Health: {plan['current_health_score']}")
print(f"Target Health: {plan['target_health_score']}")
print(f"Estimated Timeline: {plan['estimated_timeline_days']} days")

print("Optimization Steps:")
for i, step in enumerate(plan['optimization_steps']):
    print(f"{i+1}. {step['description']}")
    print(f"   Priority: {step['priority']}")
    print(f"   Effort: {step['effort_level']}")
    print(f"   Estimated Impact: {step['estimated_impact']}")
```

## Ad Sentiment Analyzer API

The Ad Sentiment Analyzer evaluates the sentiment and emotional aspects of ad content.

### Analyze Sentiment

```python
# Basic sentiment analysis
sentiment = client.sentiment.analyze(
    ad_content="Experience the ultimate comfort with our premium mattress.",
    platform="facebook"
)

print(f"Sentiment: {sentiment['sentiment']}")
print(f"Sentiment Score: {sentiment['sentiment_score']:.2f}")
print(f"Confidence: {sentiment['confidence']:.2f}")
```

### Detailed Emotion Analysis

```python
# Get detailed emotion analysis
emotions = client.sentiment.analyze_emotions(
    ad_content="Experience the ultimate comfort with our premium mattress.",
    platform="facebook",
    detailed=True
)

print(f"Primary Emotion: {emotions['primary_emotion']}")
print(f"Emotion Intensity: {emotions['emotion_intensity']:.2f}")

print("Emotion Breakdown:")
for emotion, score in emotions['emotion_scores'].items():
    print(f"{emotion}: {score:.2f}")
```

### Sentiment and Audience Analysis

```python
# Analyze sentiment targeting specific audiences
audience_sentiment = client.sentiment.analyze_for_audience(
    ad_content="Experience the ultimate comfort with our premium mattress.",
    platform="facebook",
    target_audiences=["general", "millennials", "seniors"]
)

print("Sentiment by Audience:")
for audience, result in audience_sentiment['audience_results'].items():
    print(f"{audience}:")
    print(f"  Sentiment: {result['sentiment']}")
    print(f"  Score: {result['sentiment_score']:.2f}")
    print(f"  Primary Emotion: {result['primary_emotion']}")
```

### Batch Sentiment Analysis

```python
# Analyze multiple ad contents
ad_contents = [
    "Experience the ultimate comfort with our premium mattress.",
    "Sleep better tonight with the most comfortable mattress ever made.",
    "Our mattress: Scientifically designed for the perfect night's sleep."
]

batch_results = client.sentiment.batch_analyze(
    ad_contents=ad_contents,
    platform="facebook"
)

for i, result in enumerate(batch_results['results']):
    print(f"Ad {i+1}:")
    print(f"  Sentiment: {result['sentiment']}")
    print(f"  Score: {result['sentiment_score']:.2f}")
    print(f"  Primary Emotion: {result['primary_emotion']}")
```

## Batch Processing

For large-scale predictions, use the batch processing capabilities.

### Batch Ad Score Prediction

```python
import pandas as pd

# Prepare batch data
ads_df = pd.DataFrame({
    'ad_id': ['ad1', 'ad2', 'ad3'],
    'ad_content': [
        'Experience the ultimate comfort with our premium mattress.',
        'Sleep better tonight with the most comfortable mattress ever made.',
        'Our mattress: Scientifically designed for the perfect night's sleep.'
    ],
    'platform': ['facebook', 'facebook', 'facebook'],
    'ad_format': ['image', 'image', 'image']
})

# Submit batch job
batch_job = client.ad_score.predict_batch(
    ads_data=ads_df,
    industry="home_goods",
    notification_email="user@example.com"  # Optional
)

print(f"Batch Job ID: {batch_job['job_id']}")
print(f"Status: {batch_job['status']}")
print(f"Estimated completion time: {batch_job['estimated_completion_time']}")

# Check batch job status
job_status = client.get_batch_job_status(job_id=batch_job['job_id'])
print(f"Current Status: {job_status['status']}")
print(f"Progress: {job_status['progress_percentage']}%")

# Get batch results when complete
if job_status['status'] == 'completed':
    results = client.get_batch_job_results(job_id=batch_job['job_id'])
    results_df = pd.DataFrame(results['results'])
    print(results_df.head())
```

### Asynchronous Processing with Callbacks

```python
# Define callback function
def on_completion(results):
    print(f"Job completed with {len(results['results'])} results")
    results_df = pd.DataFrame(results['results'])
    results_df.to_csv('ad_scores.csv', index=False)

# Submit asynchronous job with callback
client.ad_score.predict_batch_async(
    ads_data=ads_df,
    industry="home_goods",
    on_completion=on_completion
)
```

## Error Handling

The SDK provides comprehensive error handling.

```python
from within import Client
from within.exceptions import (
    WithinAPIError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    NotFoundError,
    ServerError
)

client = Client(api_key="your_api_key")

try:
    result = client.ad_score.predict(
        ad_content="Experience the ultimate comfort with our premium mattress.",
        platform="facebook",
        ad_format="image",
        industry="home_goods"
    )
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
except ValidationError as e:
    print(f"Invalid input parameters: {e}")
    print(f"Validation details: {e.validation_errors}")
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
    print(f"Reset at: {e.reset_time}")
except NotFoundError as e:
    print(f"Resource not found: {e}")
except ServerError as e:
    print(f"Server error: {e}")
except WithinAPIError as e:
    print(f"API error: {e}")
    print(f"Error code: {e.error_code}")
    print(f"Request ID: {e.request_id}")
```

## Rate Limiting

The SDK handles rate limiting automatically with configurable retry behavior.

```python
from within import Client
from within.retry import RetryStrategy

# Configure custom retry strategy
retry_strategy = RetryStrategy(
    max_retries=5,
    retry_codes=[429, 500, 502, 503, 504],
    backoff_factor=0.5,
    jitter=True
)

client = Client(
    api_key="your_api_key",
    retry_strategy=retry_strategy
)
```

## Advanced Configuration

### Proxy Configuration

```python
from within import Client

client = Client(
    api_key="your_api_key",
    proxies={
        'http': 'http://proxy.example.com:8080',
        'https': 'https://proxy.example.com:8080'
    }
)
```

### Custom API Endpoint

```python
from within import Client

client = Client(
    api_key="your_api_key",
    base_url="https://custom-api.within.co/v2"
)
```

### Connection Pooling

```python
from within import Client

client = Client(
    api_key="your_api_key",
    pool_connections=10,
    pool_maxsize=20
)
```

## Logging and Debugging

The SDK uses Python's standard logging module.

```python
import logging
from within import Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('within')
logger.setLevel(logging.DEBUG)

# Add handler for detailed request logging
handler = logging.FileHandler('within_api.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Initialize client with debug mode
client = Client(
    api_key="your_api_key",
    debug=True
)
```

## Examples

### Ad Score Predictor Examples

#### Competitive Ad Analysis

```python
# Compare your ad against a set of competitor ads
your_ad = "Our innovative standing desk adjusts to any height with the touch of a button."
competitor_ads = [
    "Standing desk with electric height adjustment for maximum comfort.",
    "Ergonomic standing desks with whisper-quiet motors and memory settings."
]

# Analyze your ad
your_score = client.ad_score.predict(
    ad_content=your_ad,
    platform="facebook",
    ad_format="image",
    industry="office_supplies"
)

# Analyze competitor ads
competitor_scores = []
for ad in competitor_ads:
    score = client.ad_score.predict(
        ad_content=ad,
        platform="facebook",
        ad_format="image",
        industry="office_supplies"
    )
    competitor_scores.append(score)

# Compare results
print(f"Your Ad Score: {your_score['overall_score']}")
print("Competitor Scores:")
for i, score in enumerate(competitor_scores):
    print(f"Competitor {i+1}: {score['overall_score']}")

# Calculate percentile against competitors
all_scores = [your_score['overall_score']] + [s['overall_score'] for s in competitor_scores]
percentile = sum(your_score['overall_score'] >= s for s in all_scores) / len(all_scores) * 100
print(f"Your ad is in the {percentile:.1f}th percentile of analyzed ads")
```

#### Ad Optimization Workflow

```python
# Start with initial ad
initial_ad = "Our mattress provides great comfort for better sleep."
platform = "facebook"
ad_format = "image"
industry = "home_goods"

# Get initial score
initial_score = client.ad_score.predict(
    ad_content=initial_ad,
    platform=platform,
    ad_format=ad_format,
    industry=industry
)
print(f"Initial Score: {initial_score['overall_score']}")

# Get optimization suggestions
suggestions = client.ad_score.get_suggestions(
    ad_content=initial_ad,
    platform=platform,
    ad_format=ad_format,
    industry=industry
)

print("Optimization Suggestions:")
for i, suggestion in enumerate(suggestions['suggestions']):
    print(f"{i+1}. {suggestion['description']}")

# Apply suggestions and generate improved versions
improved_versions = [
    "Experience unparalleled comfort and wake up refreshed with our premium memory foam mattress.",
    "Transform your sleep quality tonight with our ergonomically designed mattress, backed by sleep scientists.",
    "Our mattress combines cloud-like comfort with proper spine alignment, giving you the best sleep of your life."
]

# Score improved versions
improved_scores = []
for version in improved_versions:
    score = client.ad_score.predict(
        ad_content=version,
        platform=platform,
        ad_format=ad_format,
        industry=industry
    )
    improved_scores.append((version, score['overall_score']))

# Find the best version
improved_scores.sort(key=lambda x: x[1], reverse=True)
best_version, best_score = improved_scores[0]

print(f"Best Version (Score: {best_score}):")
print(best_version)
print(f"Improvement: +{best_score - initial_score['overall_score']:.1f} points")
```

### Account Health Predictor Examples

#### Weekly Account Health Monitoring

```python
import pandas as pd
from datetime import datetime, timedelta

# Get last 90 days of account data
end_date = datetime.now().date()
start_date = end_date - timedelta(days=90)

# Fetch metrics data (replace with your actual data fetch logic)
metrics_data = pd.DataFrame({
    'date': pd.date_range(start=start_date, end=end_date, freq='D'),
    'impressions': [...],  # Daily impressions
    'clicks': [...],       # Daily clicks
    'conversions': [...],  # Daily conversions
    'spend': [...]         # Daily spend
})

# Monitor weekly account health
account_id = "123456789"
platform = "facebook"

results = []
for week_end in pd.date_range(start=start_date + timedelta(days=6), end=end_date, freq='7D'):
    week_start = week_end - timedelta(days=6)
    week_data = metrics_data[(metrics_data['date'] >= pd.Timestamp(week_start)) & 
                             (metrics_data['date'] <= pd.Timestamp(week_end))]
    
    # Get account health for the week
    health = client.account_health.predict(
        account_id=account_id,
        platform=platform,
        metrics_data=week_data
    )
    
    results.append({
        'week_ending': week_end.strftime('%Y-%m-%d'),
        'health_score': health['health_score'],
        'risk_level': health['risk_level'],
        'trend': health['trend_direction']
    })

# Convert to DataFrame for analysis
health_trends = pd.DataFrame(results)
print(health_trends)

# Plot health score trend
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(health_trends['week_ending'], health_trends['health_score'], marker='o')
plt.title('Weekly Account Health Score Trend')
plt.xlabel('Week Ending')
plt.ylabel('Health Score')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

#### Multi-Account Health Assessment

```python
import pandas as pd

# Define account IDs and platforms
accounts = [
    {"id": "123456789", "platform": "facebook", "name": "Main FB Account"},
    {"id": "987654321", "platform": "google", "name": "Google Ads"},
    {"id": "456789123", "platform": "tiktok", "name": "TikTok Ads"}
]

# Get health assessments for all accounts
health_results = []

for account in accounts:
    # Fetch metrics data (replace with your actual data fetch logic)
    metrics_data = get_account_metrics(account["id"], account["platform"])
    
    # Get account health
    health = client.account_health.predict(
        account_id=account["id"],
        platform=account["platform"],
        metrics_data=metrics_data
    )
    
    health_results.append({
        "account_name": account["name"],
        "account_id": account["id"],
        "platform": account["platform"],
        "health_score": health["health_score"],
        "risk_level": health["risk_level"],
        "trend": health["trend_direction"]
    })

# Create DataFrame for comparison
health_df = pd.DataFrame(health_results)
print(health_df)

# Identify accounts that need attention (health score < 70)
at_risk_accounts = health_df[health_df["health_score"] < 70]
if not at_risk_accounts.empty:
    print("\nAccounts Requiring Attention:")
    for _, account in at_risk_accounts.iterrows():
        print(f"{account['account_name']} ({account['platform']}): Score {account['health_score']}, Risk Level: {account['risk_level']}")
        
        # Get detailed recommendations for at-risk accounts
        detailed_health = client.account_health.predict(
            account_id=account["account_id"],
            platform=account["platform"],
            metrics_data=get_account_metrics(account["account_id"], account["platform"]),
            include_recommendations=True
        )
        
        print("  Top Recommendations:")
        for i, rec in enumerate(detailed_health["recommendations"][:3]):
            print(f"  {i+1}. {rec['description']} (Priority: {rec['priority']})")
```

### Ad Sentiment Analyzer Examples

#### Ad Copy Sentiment Optimization

```python
# Analyze multiple ad copy variants for sentiment
ad_variants = [
    "Don't miss out on this limited-time offer!",
    "Join thousands of satisfied customers today!",
    "Transform your life with our revolutionary product.",
    "Struggling with productivity? Our solution helps you do more in less time.",
    "We've solved the biggest problems that professionals face daily."
]

results = []
for variant in ad_variants:
    # Analyze sentiment
    sentiment = client.sentiment.analyze(
        ad_content=variant,
        platform="facebook"
    )
    
    # Analyze emotions
    emotions = client.sentiment.analyze_emotions(
        ad_content=variant,
        platform="facebook"
    )
    
    results.append({
        "ad_copy": variant,
        "sentiment": sentiment["sentiment"],
        "sentiment_score": sentiment["sentiment_score"],
        "primary_emotion": emotions["primary_emotion"],
        "emotion_intensity": emotions["emotion_intensity"]
    })

# Convert to DataFrame
import pandas as pd
results_df = pd.DataFrame(results)

# Find the most positive ad copy
most_positive = results_df.loc[results_df["sentiment_score"].idxmax()]
print(f"Most Positive Ad Copy (Score: {most_positive['sentiment_score']:.2f}):")
print(most_positive["ad_copy"])
print(f"Primary Emotion: {most_positive['primary_emotion']}")

# Also find variants with specific emotions
joy_variants = results_df[results_df["primary_emotion"] == "joy"]
if not joy_variants.empty:
    print("\nVariants Evoking Joy:")
    for _, variant in joy_variants.iterrows():
        print(f"- {variant['ad_copy']} (Intensity: {variant['emotion_intensity']:.2f})")
```

#### Audience Sentiment Analysis

```python
# Define ad copy and target audiences
ad_copy = "Our innovative solution helps you save time and reduce stress."
audiences = ["general", "millennials", "senior_citizens", "parents", "professionals"]

# Analyze sentiment for each audience
audience_results = []
for audience in audiences:
    result = client.sentiment.analyze_for_audience(
        ad_content=ad_copy,
        platform="facebook",
        target_audience=audience
    )
    
    audience_results.append({
        "audience": audience,
        "sentiment": result["sentiment"],
        "sentiment_score": result["sentiment_score"],
        "primary_emotion": result["primary_emotion"]
    })

# Convert to DataFrame
import pandas as pd
audience_df = pd.DataFrame(audience_results)
print(audience_df)

# Identify best-matching audiences
best_audiences = audience_df.sort_values("sentiment_score", ascending=False).head(2)
print("\nBest Target Audiences:")
for _, audience in best_audiences.iterrows():
    print(f"{audience['audience']}: Score {audience['sentiment_score']:.2f}, Primary Emotion: {audience['primary_emotion']}")

# Identify audiences needing copy adjustments
low_score_audiences = audience_df[audience_df["sentiment_score"] < 0.6]
if not low_score_audiences.empty:
    print("\nAudiences Needing Copy Adjustments:")
    for _, audience in low_score_audiences.iterrows():
        print(f"{audience['audience']}: Score {audience['sentiment_score']:.2f}, Primary Emotion: {audience['primary_emotion']}")
```

## Changelog

### Version 2.1.0 (2023-06-15)
- Added support for Ad Sentiment Analyzer model
- Improved batch processing with progress tracking
- Enhanced error handling and retry logic
- Added audience-specific sentiment analysis

### Version 2.0.0 (2023-03-01)
- Major API redesign with unified client interface
- Added Account Health Predictor integration
- Improved documentation and examples
- Added asynchronous batch processing
- Enhanced error reporting and validation

### Version 1.5.0 (2022-10-10)
- Added ad comparison functionality
- Improved performance for batch operations
- Enhanced logging and debugging options
- Added proxy support

### Version 1.0.0 (2022-05-01)
- Initial release with Ad Score Predictor
- Basic prediction and optimization functionality
- Authentication and error handling 