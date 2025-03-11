# Sentiment Analysis Integration Guide

**IMPLEMENTATION STATUS: IMPLEMENTED**


This document provides detailed guidance on integrating the WITHIN Ad Sentiment Analyzer into your applications, workflows, and platforms.

## Table of Contents

- [Overview](#overview)
- [Integration Options](#integration-options)
- [API Integration](#api-integration)
- [SDK Integration](#sdk-integration)
- [Batch Processing](#batch-processing)
- [Integration with Advertising Platforms](#integration-with-advertising-platforms)
- [Usage in Decision Workflows](#usage-in-decision-workflows)
- [Custom Integrations](#custom-integrations)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [Example Use Cases](#example-use-cases)
- [Advanced Configurations](#advanced-configurations)

## Overview

The WITHIN Ad Sentiment Analyzer provides powerful sentiment and emotion analysis capabilities specifically designed for advertising content. This guide explains how to integrate these capabilities into your applications and workflows.

### Key Integration Benefits

- **Improve Ad Performance**: Optimize ad messaging for better engagement
- **Automate Content Reviews**: Automatically screen ad copy for sentiment issues
- **Enhance Reporting**: Add sentiment dimensions to performance reporting
- **Inform Content Strategy**: Gain insights for content development
- **A/B Testing**: Compare sentiment profiles of different ad variants

### Integration Considerations

Before integrating, consider these key factors:

- **Volume of Analysis**: Number of texts to analyze per day/hour
- **Response Time Requirements**: Latency constraints for your application
- **Integration Pattern**: Real-time API, batch processing, or SDK
- **Data Flow**: How ad content reaches the analyzer and how results are used
- **Security and Privacy**: Requirements for data handling
- **Context Requirements**: Need for industry, platform, or audience-specific calibration 

## Integration Options

The WITHIN Ad Sentiment Analyzer offers multiple integration approaches to fit different use cases:

| Integration Method | Best For | Considerations |
|-------------------|----------|---------------|
| REST API | Web applications, services with diverse tech stacks | Requires network connectivity, potential latency |
| Python SDK | Python applications, data science workflows | Simplest for Python environments, client-side processing |
| Batch Processing | High volume analysis, ETL workflows | Most efficient for large datasets, not real-time |
| Webhooks | Event-driven architectures | Good for asynchronous workflows |
| Custom Deployment | High-security environments, edge computing | Requires more setup and maintenance |

## API Integration

The REST API provides a straightforward way to integrate sentiment analysis into any application that can make HTTP requests.

### Authentication

The API uses API key authentication:

```python
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
    "X-Within-Client-Id": client_id
}
```

API keys can be obtained from the WITHIN dashboard under API Settings.

### Basic API Request

Here's a simple example of calling the sentiment analysis API:

```python
import requests
import json

def analyze_sentiment(text, api_key, context=None):
    """Analyze ad text sentiment using the WITHIN API."""
    url = "https://api.within.co/api/v1/analyze/sentiment"
    
    # Set up the request payload
    payload = {
        "text": text,
        "context": context or {
            "industry": "general",
            "platform": "facebook"
        },
        "settings": {
            "include_aspects": True,
            "include_emotion": True
        }
    }
    
    # Set up headers with authentication
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Make the API call
    response = requests.post(url, json=payload, headers=headers)
    
    # Handle the response
    if response.status_code == 200:
        return response.json()
    else:
        error_message = f"API Error: {response.status_code}"
        try:
            error_data = response.json()
            if "error" in error_data:
                error_message += f", {error_data['error']['message']}"
        except:
            error_message += f", {response.text}"
        
        raise Exception(error_message)

# Example usage
api_key = "your_api_key_here"
ad_text = "Experience amazing results with our revolutionary product! Limited time offer."

try:
    result = analyze_sentiment(ad_text, api_key, {"industry": "retail"})
    print(f"Overall sentiment: {result['sentiment']['overall']}")
    print(f"Confidence: {result['confidence']}")
    
    # Print emotion scores
    for emotion, score in result['emotions'].items():
        if score > 0.2:  # Only show significant emotions
            print(f"{emotion}: {score:.2f}")
            
    # Print aspect sentiment
    for aspect, data in result['aspects'].items():
        print(f"Aspect '{aspect}': {data['sentiment']:.2f}")
except Exception as e:
    print(f"Error analyzing sentiment: {e}")
```

### API Response Format

The API returns a JSON response with the following structure:

```json
{
  "sentiment": {
    "overall": 0.78,
    "positive": 0.85,
    "negative": 0.07,
    "neutral": 0.08
  },
  "confidence": 0.92,
  "emotions": {
    "joy": 0.65,
    "trust": 0.72,
    "fear": 0.03,
    "surprise": 0.45,
    "sadness": 0.04,
    "anger": 0.02,
    "anticipation": 0.58,
    "disgust": 0.01
  },
  "aspects": {
    "product": {
      "sentiment": 0.82,
      "confidence": 0.88,
      "spans": ["revolutionary product"]
    },
    "offer": {
      "sentiment": 0.75,
      "confidence": 0.91,
      "spans": ["Limited time offer"]
    }
  },
  "language": "en",
  "request_id": "req_1234567890"
}
```

### API Endpoints

| Endpoint | Description | Documentation |
|----------|-------------|---------------|
| `/api/v1/analyze/sentiment` | Analyze sentiment of a single text | [API Reference](/docs/api/endpoints.md#sentiment-analysis) |
| `/api/v1/analyze/sentiment/batch` | Batch analysis of multiple texts | [API Reference](/docs/api/endpoints.md#batch-sentiment-analysis) |
| `/api/v1/analyze/emotions` | Detailed emotion analysis | [API Reference](/docs/api/endpoints.md#emotion-analysis) |

### API Versioning

The API uses a versioned URL structure (`/api/v1/...`). When breaking changes are introduced, a new version will be released while maintaining backward compatibility with the previous version for at least 12 months.

## SDK Integration

The WITHIN Python SDK provides a more convenient integration option for Python applications.

### Installation

```bash
pip install within-sdk
```

### Basic SDK Usage

```python
from within import Client
import pandas as pd

# Initialize the client
client = Client(api_key="your_api_key_here")

# Analyze a single ad
result = client.analyze_sentiment(
    text="Discover the perfect solution for your needs. Try now risk-free!",
    context={
        "industry": "software",
        "platform": "google",
        "audience": {"demographic": "professionals"}
    }
)

# Print the results
print(f"Overall sentiment: {result.sentiment.overall}")
print(f"Confidence: {result.confidence}")

# Access emotions
for emotion_name, score in result.emotions.items():
    if score > 0.2:
        print(f"{emotion_name}: {score:.2f}")

# Access aspect-based sentiment
for aspect_name, aspect_data in result.aspects.items():
    print(f"{aspect_name}: {aspect_data.sentiment:.2f}")
```

### Analyzing Multiple Ads

The SDK provides efficient methods for analyzing multiple ads:

```python
# Create a DataFrame with ads
ads_df = pd.DataFrame({
    "ad_id": ["ad1", "ad2", "ad3"],
    "ad_text": [
        "Limited time offer! Save 50% on all products.",
        "The most advanced features for serious professionals.",
        "Don't miss out on this amazing opportunity!"
    ],
    "platform": ["facebook", "linkedin", "instagram"]
})

# Analyze all ads
results = client.analyze_sentiment_batch(
    texts=ads_df["ad_text"].tolist(),
    contexts=[
        {"platform": platform, "industry": "retail"}
        for platform in ads_df["platform"]
    ],
    ids=ads_df["ad_id"].tolist()
)

# Add results to the DataFrame
sentiment_scores = [r.sentiment.overall for r in results]
confidence_scores = [r.confidence for r in results]

ads_df["sentiment_score"] = sentiment_scores
ads_df["confidence"] = confidence_scores

# Print the augmented DataFrame
print(ads_df)
```

## Batch Processing

For large-scale analysis, batch processing provides the most efficient approach.

### Batch API Endpoint

The batch API endpoint allows processing multiple texts in a single request:

```python
import requests
import json
import time

def batch_analyze_sentiment(texts, api_key, contexts=None, settings=None):
    """Analyze sentiment for multiple ad texts in batch."""
    url = "https://api.within.co/api/v1/analyze/sentiment/batch"
    
    # Create batch items
    items = []
    for i, text in enumerate(texts):
        item = {
            "id": str(i),
            "text": text
        }
        
        # Add context if provided
        if contexts and i < len(contexts):
            item["context"] = contexts[i]
            
        items.append(item)
    
    # Create payload
    payload = {
        "items": items,
        "settings": settings or {
            "include_aspects": True,
            "include_emotions": True
        }
    }
    
    # Set headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Make request
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 202:  # Accepted for processing
        # Get batch job ID
        job_id = response.json().get("job_id")
        print(f"Batch job submitted with ID: {job_id}")
        
        # Poll for results
        return poll_batch_results(job_id, api_key)
    elif response.status_code == 200:  # Immediate response
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code}, {response.text}")

def poll_batch_results(job_id, api_key, max_attempts=30, interval=2):
    """Poll for batch job results."""
    url = f"https://api.within.co/api/v1/jobs/{job_id}"
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    for attempt in range(max_attempts):
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            job_status = response.json()
            
            if job_status["status"] == "completed":
                return job_status["results"]
            elif job_status["status"] == "failed":
                raise Exception(f"Batch job failed: {job_status.get('error')}")
            
            # Job still processing
            print(f"Job {job_id} progress: {job_status.get('progress', 0):.0f}%")
        
        # Wait before next attempt
        time.sleep(interval)
    
    raise Exception(f"Timed out waiting for batch job {job_id}")

# Example usage for large dataset
api_key = "your_api_key_here"

# Read ad texts from CSV
import pandas as pd
ads = pd.read_csv("ads_dataset.csv")

# Process in batches of 100
batch_size = 100
for i in range(0, len(ads), batch_size):
    batch = ads.iloc[i:i+batch_size]
    
    try:
        # Process batch
        results = batch_analyze_sentiment(
            texts=batch["ad_text"].tolist(),
            contexts=[{"platform": p} for p in batch["platform"]],
            api_key=api_key
        )
        
        # Store results in DataFrame
        for j, result in enumerate(results):
            index = i + j
            if index < len(ads):
                ads.at[index, "sentiment_score"] = result["sentiment"]["overall"]
                ads.at[index, "primary_emotion"] = max(
                    result["emotions"].items(),
                    key=lambda x: x[1]
                )[0]
        
        print(f"Processed batch {i//batch_size + 1}/{(len(ads) + batch_size - 1)//batch_size}")
    
    except Exception as e:
        print(f"Error processing batch starting at index {i}: {e}")

# Save results
ads.to_csv("ads_with_sentiment.csv", index=False)
```

### Asynchronous Batch Processing

For very large datasets, WITHIN provides an asynchronous batch processing facility:

1. Upload a CSV file with ad texts to analyze
2. Start an asynchronous batch job
3. Receive a notification when processing is complete
4. Download results

```python
from within import Client

# Initialize client
client = Client(api_key="your_api_key_here")

# Submit asynchronous batch job
job = client.submit_async_batch(
    file_path="large_ads_dataset.csv",
    text_column="ad_text",
    context_columns={
        "platform": "platform",
        "industry": "category"
    },
    id_column="ad_id",
    notification_email="user@example.com"
)

print(f"Submitted job {job.id}, current status: {job.status}")

# Check job status (can be called later)
job_status = client.get_job_status(job.id)
print(f"Job progress: {job_status.progress}%")

# When job is complete, download results
if job_status.status == "completed":
    client.download_job_results(
        job_id=job.id,
        output_path="sentiment_results.csv"
    )
```

### Scheduling Regular Batch Jobs

For recurring analysis, you can set up scheduled batch jobs:

```python
from within import Client
from datetime import datetime, timedelta

# Initialize client
client = Client(api_key="your_api_key_here")

# Schedule a weekly batch job
schedule = client.schedule_batch_job(
    name="Weekly Ad Sentiment Analysis",
    data_source={
        "type": "database",
        "connection_string": "${DATABASE_CONNECTION}",
        "query": "SELECT id, ad_text, platform FROM ads WHERE created_at > NOW() - INTERVAL '7 days'"
    },
    schedule={
        "frequency": "weekly",
        "day_of_week": "monday",
        "time": "02:00:00",
        "timezone": "UTC"
    },
    output={
        "type": "s3",
        "bucket": "my-company-analytics",
        "path": "ad-sentiment/{date}/results.csv"
    },
    notifications=["analytics-team@example.com"]
)

print(f"Scheduled job {schedule.id}, next run: {schedule.next_run}")
```

## Integration with Advertising Platforms

The WITHIN Ad Sentiment Analyzer can be integrated with major advertising platforms to analyze ad content before publishing.

### Facebook Ads Integration

```python
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.api import FacebookAdsApi
from within import Client

# Set up Facebook API client
FacebookAdsApi.init(access_token='your_facebook_access_token')
account = AdAccount('act_<your_ad_account_id>')

# Set up WITHIN client
within_client = Client(api_key="your_within_api_key")

# Get draft ads
ads = account.get_ads(params={
    'status': ['PAUSED', 'DRAFT'],
    'fields': ['id', 'creative', 'status']
})

# Analyze ad creatives
for ad in ads:
    try:
        # Get creative details
        creative_id = ad['creative']['id']
        creative = ad.api_get(params={'fields': ['object_story_spec']})
        
        # Extract ad text
        ad_text = creative['object_story_spec']['link_data']['message']
        
        # Analyze sentiment
        result = within_client.analyze_sentiment(
            text=ad_text,
            context={"platform": "facebook"}
        )
        
        # Log results
        print(f"Ad {ad['id']}: Sentiment Score {result.sentiment.overall:.2f}")
        
        # Optional: Make decisions based on sentiment
        if result.sentiment.overall < 0.3:
            print(f"⚠️ Ad {ad['id']} has low sentiment score, review recommended")
            
        # Flag problematic emotions if present
        if result.emotions.get('anger', 0) > 0.4 or result.emotions.get('fear', 0) > 0.6:
            print(f"⚠️ Ad {ad['id']} contains strong negative emotions")
            
    except Exception as e:
        print(f"Error analyzing ad {ad['id']}: {e}")
```

### Google Ads Integration

```python
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.v11.services.types import google_ads_service
from within import Client

# Set up Google Ads client
google_ads_client = GoogleAdsClient.load_from_storage('google-ads.yaml')

# Set up WITHIN client
within_client = Client(api_key="your_within_api_key")

# Create query to get ad text assets
query = """
    SELECT 
        ad_group_ad.ad.id, 
        ad_group_ad.ad.text_ad.headline, 
        ad_group_ad.ad.text_ad.description_1,
        ad_group_ad.ad.text_ad.description_2
    FROM ad_group_ad
    WHERE ad_group_ad.status = 'ENABLED'
"""

# Issue a search request
search_request = google_ads_service.SearchGoogleAdsRequest(
    customer_id='your_customer_id',
    query=query
)

# Get the results
search_response = google_ads_client.service.google_ads.search(request=search_request)

# Analyze ad text for each ad
for row in search_response:
    ad = row.ad_group_ad.ad
    
    # Combine headline and descriptions
    ad_text = f"{ad.text_ad.headline}. {ad.text_ad.description_1} {ad.text_ad.description_2}"
    
    try:
        # Analyze sentiment
        result = within_client.analyze_sentiment(
            text=ad_text,
            context={"platform": "google"}
        )
        
        # Log results
        print(f"Ad {ad.id}: Sentiment Score {result.sentiment.overall:.2f}")
        
        # Store analysis results in a database or file
        store_sentiment_analysis(ad.id, result)
        
    except Exception as e:
        print(f"Error analyzing ad {ad.id}: {e}")
```

## Usage in Decision Workflows

Sentiment analysis can be integrated into your decision workflows for content approval, optimization, and feedback.

### Content Approval Workflow

Here's how to implement a basic content approval workflow using sentiment analysis:

```python
from within import Client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment-approval")

# Initialize WITHIN client
client = Client(api_key="your_api_key_here")

def sentiment_approval_workflow(ad_text, context, thresholds=None):
    """
    Run ad text through a sentiment approval workflow.
    
    Args:
        ad_text: The ad text to analyze
        context: Context information (platform, industry, etc.)
        thresholds: Custom threshold values for approvals
        
    Returns:
        Dict with approval status and feedback
    """
    # Set default thresholds if not provided
    if thresholds is None:
        thresholds = {
            "min_overall_sentiment": 0.3,
            "max_negative_sentiment": 0.4,
            "max_anger": 0.3,
            "max_fear": 0.4,
            "max_disgust": 0.3,
            "required_confidence": 0.7
        }
    
    try:
        # Analyze sentiment
        result = client.analyze_sentiment(
            text=ad_text,
            context=context
        )
        
        # Check confidence
        if result.confidence < thresholds["required_confidence"]:
            return {
                "status": "MANUAL_REVIEW",
                "reason": "Low confidence score",
                "confidence": result.confidence,
                "sentiment": result.sentiment.overall,
                "analysis": result
            }
        
        # Check sentiment thresholds
        if result.sentiment.overall < thresholds["min_overall_sentiment"]:
            return {
                "status": "REJECTED",
                "reason": "Overall sentiment too low",
                "sentiment": result.sentiment.overall,
                "analysis": result
            }
        
        if result.sentiment.negative > thresholds["max_negative_sentiment"]:
            return {
                "status": "REJECTED",
                "reason": "Negative sentiment too high",
                "negative_sentiment": result.sentiment.negative,
                "analysis": result
            }
        
        # Check specific emotions
        emotions = result.emotions
        if emotions.get("anger", 0) > thresholds["max_anger"]:
            return {
                "status": "REJECTED",
                "reason": "Anger emotion too high",
                "anger_score": emotions.get("anger", 0),
                "analysis": result
            }
        
        if emotions.get("fear", 0) > thresholds["max_fear"]:
            return {
                "status": "REJECTED",
                "reason": "Fear emotion too high",
                "fear_score": emotions.get("fear", 0),
                "analysis": result
            }
        
        if emotions.get("disgust", 0) > thresholds["max_disgust"]:
            return {
                "status": "REJECTED",
                "reason": "Disgust emotion too high",
                "disgust_score": emotions.get("disgust", 0),
                "analysis": result
            }
        
        # All checks passed
        return {
            "status": "APPROVED",
            "sentiment": result.sentiment.overall,
            "analysis": result
        }
    
    except Exception as e:
        logger.error(f"Error in sentiment approval workflow: {e}")
        return {
            "status": "ERROR",
            "reason": str(e)
        }

# Example usage
ad_text = "Discover our amazing new product line! Perfect for all your needs."
context = {"platform": "facebook", "industry": "retail"}

approval_result = sentiment_approval_workflow(ad_text, context)
print(f"Approval status: {approval_result['status']}")

if approval_result['status'] != "APPROVED":
    print(f"Reason: {approval_result.get('reason', 'Unknown')}")
else:
    print(f"Sentiment score: {approval_result['sentiment']:.2f}")
```

### A/B Testing Workflow

Integrating sentiment analysis into A/B testing can help predict which ad variants might perform better:

```python
from within import Client
import pandas as pd
import numpy as np

# Initialize WITHIN client
client = Client(api_key="your_api_key_here")

def analyze_ad_variants(variants, context):
    """
    Analyze multiple ad variants to predict potential performance.
    
    Args:
        variants: List of ad text variants
        context: Context information for analysis
        
    Returns:
        DataFrame with analysis results and recommendations
    """
    results = []
    
    for i, text in enumerate(variants):
        # Analyze sentiment
        analysis = client.analyze_sentiment(
            text=text,
            context=context
        )
        
        # Store results
        results.append({
            "variant_id": i + 1,
            "text": text,
            "sentiment": analysis.sentiment.overall,
            "confidence": analysis.confidence,
            "positive": analysis.sentiment.positive,
            "negative": analysis.sentiment.negative,
            "neutral": analysis.sentiment.neutral,
            "dominant_emotion": max(
                analysis.emotions.items(),
                key=lambda x: x[1]
            )[0] if analysis.emotions else None,
            "dominant_emotion_score": max(
                analysis.emotions.values()
            ) if analysis.emotions else 0
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Add percentile rank for sentiment
    df["sentiment_percentile"] = df["sentiment"].rank(pct=True) * 100
    
    # Add predicted performance score (simplified model)
    # In a real scenario, this would use a more complex model trained on historical performance data
    df["predicted_performance"] = (
        df["sentiment"] * 0.6 + 
        df["positive"] * 0.3 - 
        df["negative"] * 0.4 +
        df["confidence"] * 0.1
    )
    
    # Rank by predicted performance
    df["rank"] = df["predicted_performance"].rank(ascending=False).astype(int)
    
    # Sort by rank
    df = df.sort_values("rank")
    
    return df

# Example usage
ad_variants = [
    "Limited time offer! 50% off all products this weekend only.",
    "Discover our premium products at special prices. Shop now!",
    "Don't miss out! Last chance to save big on all items.",
    "Quality products at unbeatable prices. Shop our collection today."
]

context = {"platform": "instagram", "industry": "fashion"}

analysis = analyze_ad_variants(ad_variants, context)
print("Ad Variant Analysis:")
print(analysis[["variant_id", "sentiment", "dominant_emotion", "predicted_performance", "rank"]])

# Print recommendation
best_variant = analysis[analysis["rank"] == 1].iloc[0]
print(f"\nRecommended variant ({best_variant['variant_id']}):")
print(f"Text: {best_variant['text']}")
print(f"Sentiment: {best_variant['sentiment']:.2f}")
```

### Feedback Loop Implementation

Implementing a feedback loop helps improve ad content based on sentiment analysis:

```python
from within import Client
import pandas as pd
from datetime import datetime

# Initialize client
client = Client(api_key="your_api_key_here")

class SentimentFeedbackSystem:
    """System for providing feedback on ad text based on sentiment analysis."""
    
    def __init__(self, client, industry, platform):
        self.client = client
        self.industry = industry
        self.platform = platform
        
        # Thresholds for different aspects
        self.thresholds = {
            "overall_sentiment": 0.5,
            "product_sentiment": 0.6,
            "offer_sentiment": 0.6,
            "emotions": {
                "joy": 0.4,
                "trust": 0.5,
                "anticipation": 0.4
            }
        }
    
    def analyze_and_provide_feedback(self, ad_text):
        """Analyze ad text and provide actionable feedback."""
        # Analyze sentiment
        analysis = self.client.analyze_sentiment(
            text=ad_text,
            context={
                "industry": self.industry,
                "platform": self.platform
            }
        )
        
        # Initialize feedback
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "ad_text": ad_text,
            "overall_rating": self._calculate_overall_rating(analysis),
            "sentiment_score": analysis.sentiment.overall,
            "strengths": [],
            "improvement_areas": [],
            "suggestions": []
        }
        
        # Check overall sentiment
        if analysis.sentiment.overall < self.thresholds["overall_sentiment"]:
            feedback["improvement_areas"].append("Overall sentiment is low")
            feedback["suggestions"].append("Use more positive and engaging language")
        else:
            feedback["strengths"].append("Good overall sentiment")
        
        # Check aspect sentiment
        for aspect, data in analysis.aspects.items():
            aspect_threshold = self.thresholds.get(f"{aspect}_sentiment", 0.5)
            
            if data.sentiment < aspect_threshold:
                feedback["improvement_areas"].append(f"Low sentiment for {aspect}")
                
                if aspect == "product":
                    feedback["suggestions"].append("Emphasize product benefits more clearly")
                elif aspect == "offer":
                    feedback["suggestions"].append("Make the offer more appealing or valuable")
                elif aspect == "brand":
                    feedback["suggestions"].append("Strengthen brand positioning in the ad")
            else:
                feedback["strengths"].append(f"Good sentiment for {aspect}")
        
        # Check emotions
        for emotion, threshold in self.thresholds["emotions"].items():
            emotion_score = analysis.emotions.get(emotion, 0)
            
            if emotion_score < threshold:
                feedback["improvement_areas"].append(f"Low {emotion} emotion")
                
                if emotion == "joy":
                    feedback["suggestions"].append("Add more uplifting or positive elements")
                elif emotion == "trust":
                    feedback["suggestions"].append("Include trust indicators or credibility elements")
                elif emotion == "anticipation":
                    feedback["suggestions"].append("Create more excitement about the outcome")
        
        return feedback
    
    def _calculate_overall_rating(self, analysis):
        """Calculate an overall rating on a 1-5 scale."""
        # This is a simplified rating calculation
        # In a real system, this would be more sophisticated
        sentiment = analysis.sentiment.overall
        
        if sentiment >= 0.8:
            return 5
        elif sentiment >= 0.6:
            return 4
        elif sentiment >= 0.4:
            return 3
        elif sentiment >= 0.2:
            return 2
        else:
            return 1

# Example usage
feedback_system = SentimentFeedbackSystem(
    client=client,
    industry="technology",
    platform="facebook"
)

ad_text = "Try our new software. It has many features."

feedback = feedback_system.analyze_and_provide_feedback(ad_text)

print(f"Ad Feedback (Rating: {feedback['overall_rating']}/5)")
print(f"Sentiment: {feedback['sentiment_score']:.2f}")

print("\nStrengths:")
for strength in feedback["strengths"]:
    print(f"- {strength}")

print("\nAreas for Improvement:")
for area in feedback["improvement_areas"]:
    print(f"- {area}")

print("\nSuggestions:")
for suggestion in feedback["suggestions"]:
    print(f"- {suggestion}")
```

## Custom Integrations

For specialized use cases, you may need to create custom integrations with the WITHIN Ad Sentiment Analyzer.

### On-Premise Deployment

For environments with strict security requirements or high-volume needs, on-premise deployment is available:

1. **Request On-Premise Package**: Contact WITHIN support to request the on-premise deployment package
2. **Hardware Setup**: Provision appropriate hardware (recommended: 8+ CPU cores, 16+ GB RAM, GPU optional but recommended)
3. **Installation**: Follow the installation guide provided with the package
4. **Configuration**: Set up integration with your existing systems

Example Docker Compose configuration for on-premise deployment:

```yaml
version: '3'

services:
  sentiment-analyzer:
    image: within/sentiment-analyzer:latest
    ports:
      - "8080:8080"
    volumes:
      - ./config:/app/config
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - MODEL_VERSION=v1.2.3
      - LOG_LEVEL=INFO
      - MAX_BATCH_SIZE=100
      - USE_GPU=true
      - AUTH_ENABLED=true
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
  
  sentiment-api:
    image: within/sentiment-api:latest
    ports:
      - "9000:9000"
    depends_on:
      - sentiment-analyzer
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    environment:
      - ANALYZER_URL=http://sentiment-analyzer:8080
      - AUTH_KEY=${AUTH_KEY}
      - LOG_LEVEL=INFO
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

### Custom Model Adaptation

For organizations with specific requirements, the WITHIN Ad Sentiment Analyzer can be customized:

1. **Industry-Specific Adaptation**: Fine-tune the model for your industry's specific terminology and sentiment patterns
2. **Custom Aspects**: Define custom aspects relevant to your products or services
3. **Brand-Specific Training**: Adapt the model to understand your brand voice and positioning

Contact the WITHIN team for more information on custom model adaptation.

## Performance Considerations

When integrating the WITHIN Ad Sentiment Analyzer, consider these performance factors to ensure optimal operation.

### Latency Optimization

If you require low-latency analysis:

1. **Use Batch Processing**: For multiple ads, batch processing reduces overhead
2. **Minimize Request Payload**: Include only necessary context information
3. **Response Filtering**: Request only the needed data using the `fields` parameter
4. **Regional Endpoints**: Use the endpoint in your geographic region
5. **Connection Pooling**: Maintain persistent connections for multiple requests

```python
# Example of optimized API call
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Create a session with connection pooling
session = requests.Session()

# Configure retries
retries = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[502, 503, 504]
)

# Mount the adapter
session.mount('https://', HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=100))

def analyze_sentiment_optimized(text, api_key, fields=None):
    """Optimized sentiment analysis request."""
    url = "https://api.within.co/api/v1/analyze/sentiment"
    
    # Build payload with only required fields
    payload = {
        "text": text,
        # Minimal context
        "context": {
            "platform": "general"
        }
    }
    
    # Add fields parameter to request only needed data
    if fields:
        payload["fields"] = fields
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Use session for connection pooling
    response = session.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code}, {response.text}")

# Example usage - requesting only needed fields
result = analyze_sentiment_optimized(
    "Experience the difference with our premium service!",
    "your_api_key",
    fields=["sentiment.overall", "confidence"]
)
```

### Throughput Optimization

For high-volume processing:

1. **Batch Size Tuning**: Optimal batch size is typically 50-100 items
2. **Parallel Processing**: Use multiple threads or processes for very large datasets
3. **Rate Limiting**: Be aware of your API rate limits and implement appropriate backoff
4. **Scheduled Processing**: Schedule batch jobs during off-peak hours
5. **Caching**: Cache results for frequently analyzed content

```python
import concurrent.futures
from within import Client
import pandas as pd

def process_large_dataset(file_path, api_key, batch_size=50, max_workers=4):
    """Process a large dataset with parallel batches."""
    # Load data
    data = pd.read_csv(file_path)
    
    # Initialize client
    client = Client(api_key=api_key)
    
    # List to store results
    all_results = []
    
    # Process in batches with parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit batches
        futures = []
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i+batch_size]
            future = executor.submit(
                process_batch,
                client=client,
                texts=batch["ad_text"].tolist(),
                contexts=batch["platform"].tolist()
            )
            futures.append((future, i))
        
        # Process results as they complete
        for future, start_idx in futures:
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                print(f"Completed batch starting at index {start_idx}")
            except Exception as e:
                print(f"Error processing batch starting at index {start_idx}: {e}")
    
    return all_results

def process_batch(client, texts, contexts):
    """Process a batch of texts."""
    results = []
    
    # Prepare contexts
    context_objects = [{"platform": p} for p in contexts]
    
    # Process batch
    batch_results = client.analyze_sentiment_batch(
        texts=texts,
        contexts=context_objects
    )
    
    # Format results
    for i, result in enumerate(batch_results):
        results.append({
            "text": texts[i],
            "platform": contexts[i],
            "sentiment": result.sentiment.overall,
            "confidence": result.confidence
        })
    
    return results
```

### Resource Usage

Monitor these resource metrics when integrating the sentiment analyzer:

| Resource | Typical Usage | Notes |
|----------|---------------|-------|
| API Calls | 0.5-1 call per ad | Batch processing reduces call count |
| Response Time | 150-300ms | Varies by text length and requested analysis |
| Request Size | 1-5 KB | Varies by text length and context data |
| Response Size | 2-15 KB | Depends on requested analysis details |
| Memory (SDK) | 200-500 MB | Python SDK memory footprint |

## Troubleshooting

Common integration issues and their solutions:

### API Connection Issues

**Problem**: Unable to connect to the WITHIN API

**Solutions**:
- Verify network connectivity and firewall settings
- Check that the API endpoint URL is correct
- Ensure SSL/TLS certificates are up to date
- Try a different network or use a proxy if necessary

### Authentication Errors

**Problem**: API calls fail with 401 (Unauthorized) or 403 (Forbidden)

**Solutions**:
- Verify API key is correct and active
- Check that the API key has the necessary permissions
- Ensure system clock is synchronized (for timestamp-based authentication)
- Verify your subscription is active and has sufficient quota

```python
# Verify API key is working
import requests

def verify_api_key(api_key):
    """Verify if an API key is working."""
    url = "https://api.within.co/api/v1/account"
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        print("API key is valid")
        return True
    elif response.status_code == 401:
        print("API key is invalid or expired")
        return False
    else:
        print(f"Unexpected status code: {response.status_code}")
        return False
```

### Rate Limiting

**Problem**: Receiving 429 (Too Many Requests) errors

**Solutions**:
- Implement exponential backoff and retry logic
- Reduce request frequency or batch requests
- Check rate limits in your subscription plan
- Distribute requests evenly over time

```python
import time
import random

def call_with_backoff(func, max_retries=5, initial_backoff=1):
    """Call a function with exponential backoff for retries."""
    retries = 0
    while retries <= max_retries:
        try:
            return func()
        except Exception as e:
            if "429" in str(e) and retries < max_retries:
                # Rate limited, apply exponential backoff
                sleep_time = initial_backoff * (2 ** retries) + random.uniform(0, 1)
                print(f"Rate limited. Retrying in {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                retries += 1
            else:
                # Different error or max retries exceeded
                raise
```

### Performance Issues

**Problem**: Analysis takes too long or times out

**Solutions**:
- Check text length (very long texts may take longer)
- Reduce requested analysis details
- Verify network latency isn't contributing to delays
- Use batch processing for multiple texts
- Check server load and try again later

### Unexpected Results

**Problem**: Sentiment scores don't match expectations

**Solutions**:
- Check if the context (industry, platform) is correctly specified
- Verify text is in a supported language
- Ensure special characters and formatting are properly handled
- Review ad text for ambiguity that might affect analysis
- Consider domain-specific terminology that might need custom handling

## Example Use Cases

### Content Moderation System

This example implements a content moderation system that flags potentially problematic ad content:

```python
from within import Client
import pandas as pd
from datetime import datetime

class ContentModerationSystem:
    """System for moderating ad content based on sentiment and emotion analysis."""
    
    def __init__(self, api_key):
        self.client = Client(api_key=api_key)
        
        # Define moderation rules
        self.moderation_rules = [
            {
                "name": "High negative sentiment",
                "condition": lambda result: result.sentiment.negative > 0.7,
                "flag_level": "high"
            },
            {
                "name": "High anger emotion",
                "condition": lambda result: result.emotions.get("anger", 0) > 0.6,
                "flag_level": "high"
            },
            {
                "name": "High fear emotion",
                "condition": lambda result: result.emotions.get("fear", 0) > 0.7,
                "flag_level": "high"
            },
            {
                "name": "Moderate negative sentiment",
                "condition": lambda result: result.sentiment.negative > 0.4,
                "flag_level": "medium"
            },
            {
                "name": "Low positive sentiment",
                "condition": lambda result: result.sentiment.positive < 0.3,
                "flag_level": "low"
            }
        ]
    
    def moderate_ad(self, ad_text, context=None):
        """
        Moderate ad text and flag potential issues.
        
        Args:
            ad_text: The ad text to moderate
            context: Context information (platform, industry, etc.)
            
        Returns:
            Dict with moderation results
        """
        # Default context
        context = context or {"platform": "general"}
        
        # Analyze sentiment
        result = self.client.analyze_sentiment(
            text=ad_text,
            context=context
        )
        
        # Apply moderation rules
        flags = []
        for rule in self.moderation_rules:
            if rule["condition"](result):
                flags.append({
                    "rule": rule["name"],
                    "level": rule["flag_level"]
                })
        
        # Determine overall flag level
        overall_level = "none"
        if any(flag["level"] == "high" for flag in flags):
            overall_level = "high"
        elif any(flag["level"] == "medium" for flag in flags):
            overall_level = "medium"
        elif any(flag["level"] == "low" for flag in flags):
            overall_level = "low"
        
        # Prepare result
        moderation_result = {
            "timestamp": datetime.now().isoformat(),
            "ad_text": ad_text,
            "overall_flag_level": overall_level,
            "flags": flags,
            "sentiment": {
                "overall": result.sentiment.overall,
                "positive": result.sentiment.positive,
                "negative": result.sentiment.negative,
                "neutral": result.sentiment.neutral
            },
            "emotions": result.emotions,
            "needs_review": overall_level in ["medium", "high"]
        }
        
        return moderation_result

# Example usage
moderator = ContentModerationSystem(api_key="your_api_key_here")

# Moderate an ad
ad_text = "Warning! Don't miss this opportunity or you'll regret it! Last chance to avoid missing out!"
context = {"platform": "facebook", "industry": "retail"}

result = moderator.moderate_ad(ad_text, context)

print(f"Moderation result: {result['overall_flag_level']} level flag")
print(f"Needs human review: {result['needs_review']}")

if result['flags']:
    print("\nFlags raised:")
    for flag in result['flags']:
        print(f"- {flag['rule']} ({flag['level']} level)")

print(f"\nOverall sentiment: {result['sentiment']['overall']:.2f}")
print(f"Negative sentiment: {result['sentiment']['negative']:.2f}")
```

### Campaign Performance Prediction

This example shows how to use sentiment analysis to predict campaign performance:

```python
import pandas as pd
import numpy as np
from within import Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load historical campaign data
campaigns = pd.read_csv("historical_campaigns.csv")

# Initialize WITHIN client
client = Client(api_key="your_api_key_here")

# Function to extract sentiment features
def extract_sentiment_features(ad_text, platform):
    """Extract sentiment features from ad text."""
    try:
        result = client.analyze_sentiment(
            text=ad_text,
            context={"platform": platform}
        )
        
        # Extract features
        features = {
            "sentiment_overall": result.sentiment.overall,
            "sentiment_positive": result.sentiment.positive,
            "sentiment_negative": result.sentiment.negative,
            "sentiment_neutral": result.sentiment.neutral,
            "confidence": result.confidence
        }
        
        # Add emotion features
        for emotion, score in result.emotions.items():
            features[f"emotion_{emotion}"] = score
        
        # Add aspect features
        for aspect, data in result.aspects.items():
            features[f"aspect_{aspect}"] = data.sentiment
        
        return features
    
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return None

# Extract features for all campaigns
print("Extracting sentiment features...")
sentiment_features = []

for i, campaign in campaigns.iterrows():
    features = extract_sentiment_features(
        campaign["ad_text"], 
        campaign["platform"]
    )
    
    if features:
        # Add campaign identifier
        features["campaign_id"] = campaign["campaign_id"]
        
        # Add performance metrics
        features["ctr"] = campaign["ctr"]
        features["conversion_rate"] = campaign["conversion_rate"]
        features["roas"] = campaign["roas"]
        
        sentiment_features.append(features)
    
    # Progress update
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1} campaigns")

# Convert to DataFrame
features_df = pd.DataFrame(sentiment_features)

# Prepare for modeling
X = features_df.drop(["campaign_id", "ctr", "conversion_rate", "roas"], axis=1)
y_ctr = features_df["ctr"]
y_conv = features_df["conversion_rate"]
y_roas = features_df["roas"]

# Split data
X_train, X_test, y_ctr_train, y_ctr_test = train_test_split(
    X, y_ctr, test_size=0.2, random_state=42
)

# Train CTR prediction model
print("Training CTR prediction model...")
ctr_model = RandomForestRegressor(n_estimators=100, random_state=42)
ctr_model.fit(X_train, y_ctr_train)

# Evaluate model
ctr_preds = ctr_model.predict(X_test)
ctr_mae = mean_absolute_error(y_ctr_test, ctr_preds)
print(f"CTR prediction MAE: {ctr_mae:.4f}")

# Function to predict performance for new ad text
def predict_ad_performance(ad_text, platform):
    """Predict performance metrics for new ad text."""
    # Extract sentiment features
    features = extract_sentiment_features(ad_text, platform)
    
    if not features:
        return None
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features])
    
    # Ensure all columns from training data are present
    for col in X.columns:
        if col not in features_df:
            features_df[col] = 0
    
    # Keep only columns used in training
    features_df = features_df[X.columns]
    
    # Make predictions
    predicted_ctr = ctr_model.predict(features_df)[0]
    
    # Return predictions
    return {
        "predicted_ctr": predicted_ctr,
        "sentiment": features["sentiment_overall"],
        "confidence": features["confidence"]
    }

# Example usage
new_ad_text = "Discover our premium product line. Exclusive offer for a limited time!"
platform = "facebook"

prediction = predict_ad_performance(new_ad_text, platform)
print("\nPrediction for new ad:")
print(f"Predicted CTR: {prediction['predicted_ctr']:.4f}")
print(f"Sentiment score: {prediction['sentiment']:.2f}")
```

## Advanced Configurations

### Custom Sentiment Calibration

For advanced users, the WITHIN API supports custom calibration of sentiment analysis:

```python
import requests
import json

def analyze_with_custom_calibration(text, api_key, calibration=None):
    """Analyze sentiment with custom calibration parameters."""
    url = "https://api.within.co/api/v1/analyze/sentiment"
    
    # Default calibration
    calibration = calibration or {
        "industry": {
            "baseline": 0.5,
            "positive_threshold": 0.6,
            "negative_threshold": 0.4,
            "weights": {
                "transformer_model": 0.7,
                "lexicon_model": 0.3
            }
        },
        "aspects": {
            "product": {
                "weight": 1.2
            },
            "price": {
                "weight": 0.8
            }
        },
        "emotions": {
            "joy": {
                "weight": 1.1
            },
            "trust": {
                "weight": 1.2
            }
        }
    }
    
    payload = {
        "text": text,
        "context": {
            "platform": "facebook"
        },
        "settings": {
            "calibration": calibration
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code}, {response.text}")
```

### Custom Aspect Definition

Define custom aspects specific to your business:

```python
import requests
import json

def analyze_with_custom_aspects(text, api_key, custom_aspects=None):
    """Analyze sentiment with custom aspect definitions."""
    url = "https://api.within.co/api/v1/analyze/sentiment"
    
    # Define custom aspects
    custom_aspects = custom_aspects or [
        {
            "name": "shipping",
            "keywords": ["shipping", "delivery", "arrive", "ship"],
            "patterns": ["(?i)\\b(free|fast|next[- ]day)\\s+(shipping|delivery)\\b"]
        },
        {
            "name": "customer_service",
            "keywords": ["support", "help", "service", "assistance"],
            "patterns": ["(?i)\\b24/7\\s+support\\b", "(?i)\\bcustomer\\s+service\\b"]
        },
        {
            "name": "return_policy",
            "keywords": ["return", "refund", "money back", "guarantee"],
            "patterns": ["(?i)\\b([0-9]+)[-]?day\\s+return\\b", "(?i)\\bmoney[- ]back\\s+guarantee\\b"]
        }
    ]
    
    payload = {
        "text": text,
        "context": {
            "platform": "facebook"
        },
        "settings": {
            "custom_aspects": custom_aspects
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code}, {response.text}")

# Example usage
api_key = "your_api_key_here"
ad_text = "Free next-day shipping on all orders! Our 30-day return policy gives you peace of mind, and our 24/7 customer service team is always ready to help."

try:
    result = analyze_with_custom_aspects(ad_text, api_key)
    
    print("Custom aspect sentiment:")
    for aspect, data in result["aspects"].items():
        print(f"{aspect}: {data['sentiment']:.2f}")
        if "spans" in data:
            print(f"  Spans: {', '.join(data['spans'])}")
except Exception as e:
    print(f"Error: {e}")
```

### Webhook Configuration

Set up webhooks to receive asynchronous notifications:

```python
import requests
import json
import hashlib
import hmac

# Set up a webhook
def register_webhook(api_key, webhook_url, secret):
    """Register a webhook for sentiment analysis notifications."""
    url = "https://api.within.co/api/v1/webhooks"
    
    payload = {
        "url": webhook_url,
        "events": ["sentiment.analyzed", "batch.completed"],
        "secret": secret
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 201:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code}, {response.text}")

# Verify webhook signature
def verify_webhook_signature(request_body, signature_header, webhook_secret):
    """Verify the signature of a webhook request."""
    expected_signature = hmac.new(
        webhook_secret.encode(),
        request_body.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature_header, expected_signature)

# Example Flask webhook handler
"""
from flask import Flask, request, jsonify

app = Flask(__name__)
WEBHOOK_SECRET = "your_webhook_secret"

@app.route('/webhook/sentiment', methods=['POST'])
def sentiment_webhook():
    # Verify signature
    signature = request.headers.get('X-Within-Signature')
    
    if not verify_webhook_signature(request.data.decode(), signature, WEBHOOK_SECRET):
        return jsonify({"error": "Invalid signature"}), 401
    
    # Process webhook data
    data = request.json
    
    if data['event'] == 'sentiment.analyzed':
        # Handle single sentiment analysis result
        result = data['data']
        ad_id = result.get('metadata', {}).get('ad_id')
        sentiment = result.get('sentiment', {}).get('overall')
        
        print(f"Received sentiment result for ad {ad_id}: {sentiment}")
        
        # Process result here
        
    elif data['event'] == 'batch.completed':
        # Handle batch completion
        batch_id = data['data']['batch_id']
        results_url = data['data']['results_url']
        
        print(f"Batch {batch_id} completed. Results available at: {results_url}")
        
        # Download and process results here
    
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(port=5000)
"""

# Register the webhook
api_key = "your_api_key_here"
webhook_url = "https://your-app.example.com/webhook/sentiment"
webhook_secret = "your_webhook_secret"

try:
    webhook = register_webhook(api_key, webhook_url, webhook_secret)
    print(f"Webhook registered with ID: {webhook['id']}")
except Exception as e:
    print(f"Error registering webhook: {e}")
```

For more detailed information, see the following resources:
- [API Documentation](/docs/api/overview.md)
- [Python SDK Documentation](/docs/api/python_sdk.md)
- [Ad Sentiment Analyzer Model Card](/docs/implementation/ml/model_card_ad_sentiment_analyzer.md)
- [NLP Pipeline Implementation](/docs/implementation/ml/nlp_pipeline.md)
- [Sentiment Analysis Methodology](/docs/implementation/ml/technical/sentiment_analysis.md)
- [Emotion Detection Implementation](/docs/implementation/ml/technical/emotion_detection.md) 