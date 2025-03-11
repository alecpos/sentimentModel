# Feature Engineering for Ad Performance Models

**IMPLEMENTATION STATUS: IMPLEMENTED**


This document details the feature engineering methodologies used in the WITHIN Ad Score & Account Health Predictor system, focusing on how raw data is transformed into model-ready features.

## Table of Contents

- [Overview](#overview)
- [Feature Categories](#feature-categories)
- [Text Features](#text-features)
- [Campaign Features](#campaign-features)
- [Temporal Features](#temporal-features)
- [Cross-Platform Features](#cross-platform-features)
- [Audience Features](#audience-features)
- [Performance History Features](#performance-history-features)
- [Feature Selection](#feature-selection)
- [Feature Importance](#feature-importance)
- [Implementation Details](#implementation-details)
- [Feature Pipelines](#feature-pipelines)
- [Validation and Testing](#validation-and-testing)
- [Common Issues and Solutions](#common-issues-and-solutions)

## Overview

Feature engineering is a critical component of the WITHIN ML models, transforming raw advertising data into predictive features. This process encompasses text analysis, temporal pattern extraction, cross-platform signals, and various derived metrics that enable accurate prediction of ad performance.

### Feature Engineering Philosophy

Our feature engineering approach follows these principles:

1. **Domain-Specific Knowledge**: Incorporate advertising industry expertise
2. **Multi-Modal Integration**: Combine text, numerical, categorical, and temporal data
3. **Platform-Aware Processing**: Account for platform-specific characteristics
4. **Audience Sensitivity**: Capture audience-specific patterns
5. **Temporal Awareness**: Model time-based patterns and seasonality
6. **Feature Stability**: Ensure features remain stable across data distributions
7. **Interpretability**: Create features that support model explanations

### Feature Engineering Pipeline

The feature engineering pipeline follows this high-level flow:

```
Raw Data → Preprocessing → Feature Extraction → Feature Selection → Feature Transformation → Model-Ready Features
```

Each step involves specialized processing depending on the feature type:

```
                          ┌─────────────────────┐
                          │     Raw Data        │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │    Preprocessing    │
                          └──────────┬──────────┘
                                     │
                 ┌───────────────────┼────────────────────┐
                 │                   │                    │
      ┌──────────▼──────────┐ ┌──────▼───────┐  ┌─────────▼─────────┐
      │   Text Processing   │ │   Numeric    │  │    Categorical    │
      │                     │ │  Processing  │  │     Processing    │
      └──────────┬──────────┘ └──────┬───────┘  └─────────┬─────────┘
                 │                   │                    │
                 └───────────────────┼────────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Feature Selection  │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │    Transformation   │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Model-Ready Data   │
                          └─────────────────────┘
```

## Feature Categories

The features used in WITHIN models fall into several categories:

### Primary Feature Categories

| Category | Description | Examples |
|----------|-------------|----------|
| Text Features | Derived from ad copy | Sentiment score, emotional content, keyword presence |
| Campaign Features | Ad campaign settings | Bid strategy, targeting criteria, campaign objective |
| Temporal Features | Time-based patterns | Day of week, season, time since campaign start |
| Cross-Platform Features | Data from multiple platforms | Relative performance, cross-platform patterns |
| Audience Features | Target audience attributes | Demographics, interests, behavior patterns |
| Performance History | Historical performance metrics | Previous CTR, conversion rate, engagement metrics |

### Feature Counts by Model

| Model | Text Features | Campaign Features | Temporal Features | Other Features | Total |
|-------|--------------|-------------------|-------------------|---------------|-------|
| Ad Score Predictor | 85+ | 30+ | 15+ | 40+ | 170+ |
| Account Health Predictor | 40+ | 60+ | 25+ | 80+ | 205+ |
| Ad Sentiment Analyzer | 120+ | 10+ | 5+ | 15+ | 150+ |

## Text Features

Text features are extracted from ad copy through a comprehensive natural language processing pipeline.

### Text Preprocessing

Before feature extraction, text undergoes these preprocessing steps:

1. **Cleaning**: Remove special characters and normalize whitespace
2. **Tokenization**: Split text into words, subwords, or characters
3. **Normalization**: Convert to lowercase and handle contractions
4. **Stop Word Removal**: Filter out common words with limited semantic value
5. **Stemming/Lemmatization**: Reduce words to their root forms
6. **Ad-Specific Handling**: Special processing for URLs, hashtags, emojis, and symbols

### Statistical Text Features

Basic statistical features extracted from ad text:

- Text length (characters, words, sentences)
- Average word length
- Sentence complexity
- Readability scores (Flesch-Kincaid, SMOG, Coleman-Liau)
- Punctuation frequency and patterns
- Part-of-speech distributions
- Keyword density

### Semantic Text Features

Features capturing meaning and context:

- Sentiment scores (overall, positive, negative, neutral)
- Emotion scores (joy, trust, fear, anticipation, etc.)
- Semantic embeddings (using transformer models)
- Topic distributions
- Named entity recognition
- Key phrase extraction
- Linguistic structure patterns

### Advertising-Specific Text Features

Features tailored to advertising content:

- Call-to-action presence and strength
- Offer clarity and specifics
- Price mentions and formats
- Urgency signals
- Exclusivity markers
- Social proof indicators
- Brand mention patterns
- Product feature emphasis
- Benefit articulation
- Problem-solution structure

### Implementation Example

Here's an example of extracting basic text features:

```python
import re
import numpy as np
import textstat
from collections import Counter
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_basic_text_features(text):
    """Extract basic statistical features from ad text."""
    features = {}
    
    # Clean text
    clean_text = re.sub(r'\s+', ' ', text).strip()
    
    # Basic counts
    features["char_count"] = len(clean_text)
    features["word_count"] = len(clean_text.split())
    
    # Process with spaCy
    doc = nlp(clean_text)
    
    # Sentence count
    features["sentence_count"] = len(list(doc.sents))
    
    # Average word length
    word_lengths = [len(token.text) for token in doc if not token.is_punct]
    features["avg_word_length"] = np.mean(word_lengths) if word_lengths else 0
    
    # Part-of-speech counts
    pos_counts = Counter([token.pos_ for token in doc])
    for pos, count in pos_counts.items():
        features[f"pos_{pos.lower()}"] = count
    
    # Readability scores
    features["readability_flesch"] = textstat.flesch_reading_ease(clean_text)
    features["readability_smog"] = textstat.smog_index(clean_text)
    features["readability_coleman_liau"] = textstat.coleman_liau_index(clean_text)
    
    # Punctuation features
    punct_counts = Counter([token.text for token in doc if token.is_punct])
    for punct, count in punct_counts.items():
        features[f"punct_{punct}"] = count
    
    return features
```

### Transformer-Based Features

For deeper semantic understanding, we use transformer models:

```python
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

def extract_transformer_features(text, max_length=128):
    """Extract features using a transformer model."""
    # Tokenize and prepare input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get embeddings from last hidden state
    last_hidden_state = outputs.last_hidden_state
    
    # Get CLS token embedding (sentence representation)
    cls_embedding = last_hidden_state[:, 0, :].numpy()
    
    # Create features dict with flattened embedding
    features = {}
    for i, value in enumerate(cls_embedding.flatten()):
        features[f"embedding_{i}"] = value
    
    return features
```

### Advertising-Specific Feature Extraction

Features specific to advertising content:

```python
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_ad_specific_features(text):
    """Extract advertising-specific features from ad text."""
    features = {}
    
    # Clean text
    clean_text = re.sub(r'\s+', ' ', text).strip()
    doc = nlp(clean_text)
    
    # Call to action detection
    cta_phrases = ["shop now", "buy now", "sign up", "learn more", 
                   "get started", "try free", "call today", "click here",
                   "order now", "register", "subscribe", "download"]
    
    features["has_cta"] = any(phrase in clean_text.lower() for phrase in cta_phrases)
    
    # Count imperative verbs (potential CTAs)
    imperative_count = 0
    for sent in doc.sents:
        # Check if sentence starts with a verb
        if len(sent) > 0 and sent[0].pos_ == "VERB":
            imperative_count += 1
    
    features["imperative_count"] = imperative_count
    
    # Price detection
    price_pattern = r'\$\d+(?:\.\d{2})?|\d+\s?(?:dollars|USD)'
    price_matches = re.findall(price_pattern, clean_text)
    features["has_price"] = len(price_matches) > 0
    features["price_mention_count"] = len(price_matches)
    
    # Offer detection
    offer_phrases = ["free", "discount", "% off", "save", "deal", 
                    "limited time", "exclusive", "special offer"]
    
    features["has_offer"] = any(phrase in clean_text.lower() for phrase in offer_phrases)
    
    # Urgency signals
    urgency_phrases = ["limited time", "ends soon", "last chance", 
                      "hurry", "today only", "while supplies last"]
    
    features["has_urgency"] = any(phrase in clean_text.lower() for phrase in urgency_phrases)
    
    # Exclusivity signals
    exclusivity_phrases = ["exclusive", "members only", "invitation", 
                         "selected customers", "special access"]
    
    features["has_exclusivity"] = any(phrase in clean_text.lower() for phrase in exclusivity_phrases)
    
    # Question detection
    features["question_count"] = clean_text.count('?')
    
    # Exclamation detection
    features["exclamation_count"] = clean_text.count('!')
    
    # Emoji detection
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               "]+", flags=re.UNICODE)
    
    emoji_matches = emoji_pattern.findall(clean_text)
    features["emoji_count"] = len(emoji_matches)
    
    return features
``` 

## Campaign Features

Campaign features capture the settings, configuration, and metadata of advertising campaigns.

### Campaign Configuration Features

Features derived from campaign settings:

- Campaign objective (awareness, consideration, conversion)
- Bid strategy (CPC, CPM, target ROAS, etc.)
- Budget allocation and pacing
- Start and end dates
- Campaign status (active, paused, etc.)
- Campaign type (search, display, social, etc.)
- Optimization goals

### Targeting Features

Features related to audience targeting:

- Geographic targeting (countries, regions, cities)
- Demographic targeting (age ranges, gender, etc.)
- Interest targeting categories
- Behavior targeting options
- Device targeting (mobile, desktop, tablet)
- Platform placement options
- Lookalike audience usage
- Custom audience usage

### Creative Configuration Features

Features related to ad creative settings:

- Ad format (image, video, carousel, etc.)
- Creative dimensions
- Video length (for video ads)
- Number of images (for carousel ads)
- Call-to-action button type
- Landing page category
- Display URL structure

### Implementation Example

Here's how to extract campaign configuration features:

```python
def extract_campaign_features(campaign_data):
    """Extract features from campaign configuration data."""
    features = {}
    
    # Campaign objective
    objective_mapping = {
        "AWARENESS": 0,
        "CONSIDERATION": 1, 
        "CONVERSION": 2
    }
    
    features["objective_encoded"] = objective_mapping.get(
        campaign_data.get("objective", "").upper(), -1
    )
    
    # One-hot encode objective
    for objective in objective_mapping:
        features[f"objective_{objective.lower()}"] = int(
            campaign_data.get("objective", "").upper() == objective
        )
    
    # Bid strategy
    bid_strategy_mapping = {
        "CPC": 0,
        "CPM": 1,
        "CPA": 2,
        "TARGET_ROAS": 3,
        "MAX_CONVERSIONS": 4
    }
    
    features["bid_strategy_encoded"] = bid_strategy_mapping.get(
        campaign_data.get("bid_strategy", "").upper(), -1
    )
    
    # One-hot encode bid strategy
    for strategy, _ in bid_strategy_mapping.items():
        features[f"bid_strategy_{strategy.lower()}"] = int(
            campaign_data.get("bid_strategy", "").upper() == strategy
        )
    
    # Budget features
    daily_budget = campaign_data.get("daily_budget", 0)
    features["daily_budget"] = daily_budget
    
    # Budget bracket features
    budget_brackets = [10, 50, 100, 500, 1000, 5000]
    for i, threshold in enumerate(budget_brackets):
        if i == 0:
            features[f"budget_under_{threshold}"] = int(daily_budget < threshold)
        else:
            prev_threshold = budget_brackets[i-1]
            features[f"budget_{prev_threshold}_to_{threshold}"] = int(
                prev_threshold <= daily_budget < threshold
            )
    
    features[f"budget_over_{budget_brackets[-1]}"] = int(
        daily_budget >= budget_brackets[-1]
    )
    
    # Status features
    status_mapping = {
        "ACTIVE": 0,
        "PAUSED": 1,
        "COMPLETED": 2,
        "ARCHIVED": 3
    }
    
    features["status_encoded"] = status_mapping.get(
        campaign_data.get("status", "").upper(), -1
    )
    
    # Target features
    features["target_cpa"] = campaign_data.get("target_cpa", 0)
    features["target_roas"] = campaign_data.get("target_roas", 0)
    
    # Campaign type features
    campaign_type = campaign_data.get("campaign_type", "").upper()
    campaign_types = ["SEARCH", "DISPLAY", "SOCIAL", "VIDEO", "SHOPPING"]
    
    for ctype in campaign_types:
        features[f"campaign_type_{ctype.lower()}"] = int(campaign_type == ctype)
    
    return features
```

### Targeting Feature Extraction

Extracting audience targeting features:

```python
def extract_targeting_features(targeting_data):
    """Extract features from targeting configuration."""
    features = {}
    
    # Geographic targeting features
    geo_targeting = targeting_data.get("geo_targeting", {})
    countries = geo_targeting.get("countries", [])
    
    # Country count and specific countries
    features["country_count"] = len(countries)
    
    major_countries = ["US", "GB", "CA", "AU", "DE", "FR", "JP"]
    for country in major_countries:
        features[f"targets_country_{country.lower()}"] = int(country in countries)
    
    # Region/city targeting
    features["has_region_targeting"] = int(len(geo_targeting.get("regions", [])) > 0)
    features["has_city_targeting"] = int(len(geo_targeting.get("cities", [])) > 0)
    features["geo_targeting_specificity"] = 0
    
    if features["has_city_targeting"]:
        features["geo_targeting_specificity"] = 3
    elif features["has_region_targeting"]:
        features["geo_targeting_specificity"] = 2
    elif features["country_count"] > 0:
        features["geo_targeting_specificity"] = 1
    
    # Demographic targeting
    demographic = targeting_data.get("demographic", {})
    age_ranges = demographic.get("age_ranges", [])
    genders = demographic.get("genders", [])
    
    # Age range features
    age_range_categories = ["13-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    for age_range in age_range_categories:
        features[f"targets_age_{age_range.replace('-', '_')}"] = int(age_range in age_ranges)
    
    features["age_range_count"] = len(age_ranges)
    
    # Gender targeting
    features["targets_male"] = int("MALE" in genders)
    features["targets_female"] = int("FEMALE" in genders)
    features["targets_all_genders"] = int(len(genders) == 0 or (features["targets_male"] and features["targets_female"]))
    
    # Interest targeting
    interests = targeting_data.get("interests", [])
    features["interest_count"] = len(interests)
    
    # Common interest categories
    interest_categories = [
        "technology", "fashion", "sports", "travel", "food", 
        "fitness", "entertainment", "business", "education"
    ]
    
    for category in interest_categories:
        features[f"interest_{category}"] = 0
    
    for interest in interests:
        interest_name = interest.get("name", "").lower()
        for category in interest_categories:
            if category in interest_name:
                features[f"interest_{category}"] = 1
    
    # Device targeting
    devices = targeting_data.get("devices", [])
    device_types = ["MOBILE", "DESKTOP", "TABLET"]
    
    for device in device_types:
        features[f"targets_device_{device.lower()}"] = int(device in devices)
    
    features["device_count"] = len(devices)
    features["targets_all_devices"] = int(len(devices) == len(device_types) or len(devices) == 0)
    
    # Custom audiences
    custom_audiences = targeting_data.get("custom_audiences", [])
    features["custom_audience_count"] = len(custom_audiences)
    features["has_custom_audience"] = int(features["custom_audience_count"] > 0)
    
    # Lookalike audiences
    lookalike_audiences = targeting_data.get("lookalike_audiences", [])
    features["lookalike_audience_count"] = len(lookalike_audiences)
    features["has_lookalike_audience"] = int(features["lookalike_audience_count"] > 0)
    
    return features
```

### Creative Configuration Extraction

Extracting features from ad creative configuration:

```python
def extract_creative_features(creative_data):
    """Extract features from creative configuration."""
    features = {}
    
    # Creative type
    creative_type = creative_data.get("type", "").upper()
    
    creative_types = ["IMAGE", "VIDEO", "CAROUSEL", "COLLECTION", "TEXT"]
    for ctype in creative_types:
        features[f"creative_type_{ctype.lower()}"] = int(creative_type == ctype)
    
    # Video features (for video ads)
    if creative_type == "VIDEO":
        video_data = creative_data.get("video_data", {})
        video_length = video_data.get("duration_seconds", 0)
        
        features["video_length"] = video_length
        
        # Video length brackets
        video_brackets = [15, 30, 60, 120, 300]
        for i, threshold in enumerate(video_brackets):
            if i == 0:
                features[f"video_under_{threshold}s"] = int(video_length < threshold)
            else:
                prev_threshold = video_brackets[i-1]
                features[f"video_{prev_threshold}s_to_{threshold}s"] = int(
                    prev_threshold <= video_length < threshold
                )
        
        features[f"video_over_{video_brackets[-1]}s"] = int(
            video_length >= video_brackets[-1]
        )
        
        # Video aspect ratio
        height = video_data.get("height", 0)
        width = video_data.get("width", 0)
        
        if height > 0 and width > 0:
            aspect_ratio = width / height
            features["video_aspect_ratio"] = aspect_ratio
            
            # Common aspect ratios
            features["video_square"] = int(0.9 < aspect_ratio < 1.1)
            features["video_landscape"] = int(aspect_ratio >= 1.1)
            features["video_portrait"] = int(aspect_ratio <= 0.9)
    
    # Image features (for image ads)
    if creative_type == "IMAGE":
        image_data = creative_data.get("image_data", {})
        height = image_data.get("height", 0)
        width = image_data.get("width", 0)
        
        if height > 0 and width > 0:
            aspect_ratio = width / height
            features["image_aspect_ratio"] = aspect_ratio
            
            # Common aspect ratios
            features["image_square"] = int(0.9 < aspect_ratio < 1.1)
            features["image_landscape"] = int(aspect_ratio >= 1.1)
            features["image_portrait"] = int(aspect_ratio <= 0.9)
    
    # Carousel features
    if creative_type == "CAROUSEL":
        carousel_data = creative_data.get("carousel_data", {})
        cards = carousel_data.get("cards", [])
        
        features["carousel_card_count"] = len(cards)
    
    # Call-to-action button
    cta_button = creative_data.get("cta_button", {})
    cta_type = cta_button.get("type", "").upper()
    
    cta_types = ["SHOP_NOW", "LEARN_MORE", "SIGN_UP", "CONTACT_US", "DOWNLOAD"]
    
    for cta in cta_types:
        features[f"cta_{cta.lower()}"] = int(cta_type == cta)
    
    # Landing page
    landing_page = creative_data.get("landing_page", "")
    features["has_landing_page"] = int(bool(landing_page))
    
    # Extract domain if landing page exists
    if landing_page:
        import re
        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", landing_page)
        if domain_match:
            domain = domain_match.group(1)
            features["landing_page_domain"] = domain
    
    return features
```

## Temporal Features

Temporal features capture time-based patterns and seasonality in advertising performance.

### Calendar-Based Features

Features derived from date and time:

- Day of week (0-6)
- Hour of day (0-23)
- Month of year (1-12)
- Day of month (1-31)
- Week of year (1-52)
- Quarter (1-4)
- Is weekend (boolean)
- Is holiday (boolean)
- Is business hours (boolean)

### Campaign Temporal Features

Features related to campaign timing:

- Days since campaign start
- Days until campaign end
- Campaign age bracket
- Percentage of campaign timeline elapsed
- Days since last creative refresh
- Days since last budget change
- Historical time-window aggregated metrics (7-day, 14-day, 30-day)

### Seasonality Features

Features capturing seasonal patterns:

- Season (spring, summer, fall, winter)
- Is shopping season (boolean)
- Days until major shopping events (Black Friday, Cyber Monday, etc.)
- Year-over-year seasonal patterns
- Industry-specific seasonal factors

### Implementation Example

Extracting basic temporal features:

```python
from datetime import datetime, timedelta
import holidays
import numpy as np

def extract_calendar_features(timestamp):
    """Extract features from a timestamp."""
    features = {}
    
    # Convert to datetime if string
    if isinstance(timestamp, str):
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    else:
        dt = timestamp
    
    # Basic date components
    features["day_of_week"] = dt.weekday()  # 0=Monday, 6=Sunday
    features["hour_of_day"] = dt.hour
    features["month"] = dt.month
    features["day_of_month"] = dt.day
    
    # Week of year (1-52)
    features["week_of_year"] = dt.isocalendar()[1]
    
    # Quarter (1-4)
    features["quarter"] = (dt.month - 1) // 3 + 1
    
    # Derived boolean features
    features["is_weekend"] = int(dt.weekday() >= 5)  # 5=Saturday, 6=Sunday
    
    # Business hours (9am-5pm, Monday-Friday)
    features["is_business_hours"] = int(
        0 <= dt.weekday() <= 4 and 9 <= dt.hour < 17
    )
    
    # US holidays
    us_holidays = holidays.US()
    features["is_holiday"] = int(dt.date() in us_holidays)
    
    # Season (northern hemisphere)
    month = dt.month
    if 3 <= month <= 5:
        season = "spring"
    elif 6 <= month <= 8:
        season = "summer"
    elif 9 <= month <= 11:
        season = "fall"
    else:
        season = "winter"
    
    seasons = ["winter", "spring", "summer", "fall"]
    for s in seasons:
        features[f"season_{s}"] = int(season == s)
    
    # Time of day segments
    hour = dt.hour
    if 5 <= hour < 12:
        time_segment = "morning"
    elif 12 <= hour < 17:
        time_segment = "afternoon"
    elif 17 <= hour < 21:
        time_segment = "evening"
    else:
        time_segment = "night"
    
    time_segments = ["morning", "afternoon", "evening", "night"]
    for ts in time_segments:
        features[f"time_segment_{ts}"] = int(time_segment == ts)
    
    # Cyclic encoding of time features
    # This preserves the circular nature of time features
    
    # Hour of day (0-23) -> sin and cos encoding
    features["hour_sin"] = np.sin(2 * np.pi * dt.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * dt.hour / 24)
    
    # Day of week (0-6) -> sin and cos encoding
    features["day_of_week_sin"] = np.sin(2 * np.pi * dt.weekday() / 7)
    features["day_of_week_cos"] = np.cos(2 * np.pi * dt.weekday() / 7)
    
    # Month (1-12) -> sin and cos encoding
    features["month_sin"] = np.sin(2 * np.pi * (dt.month - 1) / 12)
    features["month_cos"] = np.cos(2 * np.pi * (dt.month - 1) / 12)
    
    return features
```

### Campaign Temporal Features

Extracting campaign-specific temporal features:

```python
from datetime import datetime, timedelta

def extract_campaign_temporal_features(campaign_data, current_date=None):
    """Extract temporal features related to campaign timing."""
    features = {}
    
    # Set current date if not provided
    if current_date is None:
        current_date = datetime.now()
    elif isinstance(current_date, str):
        current_date = datetime.fromisoformat(current_date.replace('Z', '+00:00'))
    
    # Parse campaign dates
    start_date = campaign_data.get("start_date")
    end_date = campaign_data.get("end_date")
    
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
    
    if isinstance(end_date, str) and end_date:
        end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
    else:
        end_date = None
    
    # Days since campaign start
    if start_date:
        days_since_start = (current_date - start_date).days
        features["days_since_campaign_start"] = max(0, days_since_start)
        
        # Campaign age brackets
        age_brackets = [7, 30, 90, 180, 365]
        for i, threshold in enumerate(age_brackets):
            if i == 0:
                features[f"campaign_age_under_{threshold}d"] = int(days_since_start < threshold)
            else:
                prev_threshold = age_brackets[i-1]
                features[f"campaign_age_{prev_threshold}d_to_{threshold}d"] = int(
                    prev_threshold <= days_since_start < threshold
                )
        
        features[f"campaign_age_over_{age_brackets[-1]}d"] = int(
            days_since_start >= age_brackets[-1]
        )
    
    # Days until campaign end
    if end_date:
        days_until_end = (end_date - current_date).days
        features["days_until_campaign_end"] = max(0, days_until_end)
        
        # Campaign remaining time brackets
        remaining_brackets = [7, 30, 90]
        for i, threshold in enumerate(remaining_brackets):
            if i == 0:
                features[f"campaign_remaining_under_{threshold}d"] = int(days_until_end < threshold)
            else:
                prev_threshold = remaining_brackets[i-1]
                features[f"campaign_remaining_{prev_threshold}d_to_{threshold}d"] = int(
                    prev_threshold <= days_until_end < threshold
                )
        
        features[f"campaign_remaining_over_{remaining_brackets[-1]}d"] = int(
            days_until_end >= remaining_brackets[-1]
        )
    
    # Campaign progress as percentage
    if start_date and end_date:
        total_duration = (end_date - start_date).days
        if total_duration > 0:
            elapsed_duration = (current_date - start_date).days
            progress_pct = min(100, max(0, (elapsed_duration / total_duration) * 100))
            features["campaign_progress_pct"] = progress_pct
            
            # Progress brackets
            progress_brackets = [25, 50, 75]
            for i, threshold in enumerate(progress_brackets):
                if i == 0:
                    features[f"campaign_progress_under_{threshold}pct"] = int(progress_pct < threshold)
                else:
                    prev_threshold = progress_brackets[i-1]
                    features[f"campaign_progress_{prev_threshold}pct_to_{threshold}pct"] = int(
                        prev_threshold <= progress_pct < threshold
                    )
            
            features[f"campaign_progress_over_{progress_brackets[-1]}pct"] = int(
                progress_pct >= progress_brackets[-1]
            )
    
    # Last modification features
    last_modified = campaign_data.get("last_modified_date")
    if isinstance(last_modified, str):
        last_modified = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
    
    if last_modified:
        days_since_modification = (current_date - last_modified).days
        features["days_since_last_modification"] = days_since_modification
    
    # Last creative update
    last_creative_update = campaign_data.get("last_creative_update")
    if isinstance(last_creative_update, str):
        last_creative_update = datetime.fromisoformat(last_creative_update.replace('Z', '+00:00'))
    
    if last_creative_update:
        days_since_creative_update = (current_date - last_creative_update).days
        features["days_since_creative_update"] = days_since_creative_update
    
    return features
```

### Shopping Season Features

Features related to shopping seasons and holidays:

```python
from datetime import datetime, timedelta

def extract_shopping_season_features(current_date=None):
    """Extract features related to shopping seasons."""
    features = {}
    
    # Set current date if not provided
    if current_date is None:
        current_date = datetime.now()
    elif isinstance(current_date, str):
        current_date = datetime.fromisoformat(current_date.replace('Z', '+00:00'))
    
    year = current_date.year
    
    # Define major shopping events
    shopping_events = {
        "black_friday": datetime(year, 11, 1) + timedelta(days=(4 - datetime(year, 11, 1).weekday()) % 7 + 21),
        "cyber_monday": datetime(year, 11, 1) + timedelta(days=(0 - datetime(year, 11, 1).weekday()) % 7 + 28),
        "christmas": datetime(year, 12, 25),
        "valentines_day": datetime(year, 2, 14),
        "mothers_day": datetime(year, 5, 1) + timedelta(days=(6 - datetime(year, 5, 1).weekday()) % 7 + 7),
        "fathers_day": datetime(year, 6, 1) + timedelta(days=(6 - datetime(year, 6, 1).weekday()) % 7 + 14),
        "back_to_school": datetime(year, 8, 15),
        "halloween": datetime(year, 10, 31)
    }
    
    # Calculate days until/since each event
    for event_name, event_date in shopping_events.items():
        # Adjust year for past events
        if event_date < current_date:
            # If we're in Dec and the event was earlier in the year, use next year
            if current_date.month == 12 and event_date.month < 12:
                event_date = datetime(year + 1, event_date.month, event_date.day)
            # Otherwise, the event has already passed this year
            else:
                event_date = datetime(year + 1, event_date.month, event_date.day)
        
        days_until_event = (event_date - current_date).days
        features[f"days_until_{event_name}"] = days_until_event
        
        # Event proximity brackets
        proximity_brackets = [7, 14, 30, 60, 90]
        for threshold in proximity_brackets:
            features[f"{event_name}_within_{threshold}d"] = int(days_until_event <= threshold)
    
    # General shopping seasons
    
    # Holiday season (Nov 1 - Dec 31)
    is_holiday_season = (current_date.month == 11 or current_date.month == 12)
    features["is_holiday_shopping_season"] = int(is_holiday_season)
    
    # Back to school season (Jul 15 - Sep 15)
    is_back_to_school = (
        (current_date.month == 7 and current_date.day >= 15) or
        (current_date.month == 8) or
        (current_date.month == 9 and current_date.day <= 15)
    )
    features["is_back_to_school_season"] = int(is_back_to_school)
    
    # Summer season (Jun 1 - Aug 31)
    is_summer = (6 <= current_date.month <= 8)
    features["is_summer_season"] = int(is_summer)
    
    # Wedding season (May - Oct)
    is_wedding_season = (5 <= current_date.month <= 10)
    features["is_wedding_season"] = int(is_wedding_season)
    
    # Spring cleaning (Mar - May)
    is_spring_cleaning = (3 <= current_date.month <= 5)
    features["is_spring_cleaning_season"] = int(is_spring_cleaning)
    
    return features
```

## Cross-Platform Features

Cross-platform features capture relationships and patterns across multiple advertising platforms.

### Platform Comparison Features

Features that compare metrics across platforms:

- Relative CTR compared to platform average
- Relative conversion rate compared to platform average
- Relative CPC compared to platform average
- Platform-specific performance indexes
- Cross-platform performance variations

### Platform-Specific Context Features

Features that provide platform context:

- Platform-specific ad format effectiveness
- Platform benchmark metrics by industry
- Platform audience composition metrics
- Platform-specific seasonal trends
- Platform algorithm change indicators

### Implementation Example

Extracting cross-platform comparison features:

```python
import numpy as np

def extract_cross_platform_features(campaign_metrics, platform_benchmarks):
    """Extract features comparing performance across platforms."""
    features = {}
    
    # Get current platform
    platform = campaign_metrics.get("platform", "").lower()
    
    # Skip if platform not available
    if not platform or platform not in platform_benchmarks:
        return features
    
    # Get platform benchmarks
    benchmarks = platform_benchmarks[platform]
    
    # Calculate relative performance metrics
    metrics_to_compare = ["ctr", "conversion_rate", "cpc", "cpm", "roas"]
    
    for metric in metrics_to_compare:
        if metric in campaign_metrics and metric in benchmarks:
            campaign_value = campaign_metrics[metric]
            benchmark_value = benchmarks[metric]
            
            if benchmark_value > 0:
                relative_value = campaign_value / benchmark_value
                features[f"relative_{metric}"] = relative_value
                
                # Performance brackets
                if relative_value < 0.5:
                    performance_bracket = "very_low"
                elif relative_value < 0.8:
                    performance_bracket = "low"
                elif relative_value < 1.2:
                    performance_bracket = "average"
                elif relative_value < 2.0:
                    performance_bracket = "high"
                else:
                    performance_bracket = "very_high"
                
                features[f"{metric}_performance_bracket"] = performance_bracket
                
                # One-hot encode performance bracket
                brackets = ["very_low", "low", "average", "high", "very_high"]
                for bracket in brackets:
                    features[f"{metric}_{bracket}"] = int(performance_bracket == bracket)
    
    # Cross-platform performance index
    if "industry" in campaign_metrics:
        industry = campaign_metrics["industry"].lower()
        
        # Check if industry benchmarks are available
        if industry in benchmarks.get("industries", {}):
            industry_benchmarks = benchmarks["industries"][industry]
            
            for metric in metrics_to_compare:
                if metric in campaign_metrics and metric in industry_benchmarks:
                    campaign_value = campaign_metrics[metric]
                    industry_benchmark = industry_benchmarks[metric]
                    
                    if industry_benchmark > 0:
                        relative_to_industry = campaign_value / industry_benchmark
                        features[f"{metric}_vs_industry"] = relative_to_industry
    
    # Platform-specific features
    if platform == "facebook":
        features["fb_relevance_score"] = campaign_metrics.get("relevance_score", 0)
        features["fb_quality_ranking"] = campaign_metrics.get("quality_ranking", "")
        features["fb_engagement_rate_ranking"] = campaign_metrics.get("engagement_rate_ranking", "")
        
    elif platform == "google":
        features["google_quality_score"] = campaign_metrics.get("quality_score", 0)
        features["google_search_impression_share"] = campaign_metrics.get("search_impression_share", 0)
        
    elif platform == "tiktok":
        features["tiktok_video_play_rate"] = campaign_metrics.get("video_play_rate", 0)
        features["tiktok_video_watch_6s_rate"] = campaign_metrics.get("video_watch_6s_rate", 0)
    
    # Calculate competitive density
    if "competitive_density" in benchmarks:
        features["competitive_density"] = benchmarks["competitive_density"]
        
        density_brackets = ["very_low", "low", "medium", "high", "very_high"]
        density_value = features["competitive_density"]
        
        if density_value < 2:
            density_bracket = "very_low"
        elif density_value < 4:
            density_bracket = "low"
        elif density_value < 6:
            density_bracket = "medium"
        elif density_value < 8:
            density_bracket = "high"
        else:
            density_bracket = "very_high"
        
        for bracket in density_brackets:
            features[f"competitive_density_{bracket}"] = int(density_bracket == bracket)
    
    return features
```

### Platform Context Features

Extracting platform-specific context features:

```python
def extract_platform_context_features(platform, date=None):
    """Extract platform-specific context features."""
    features = {}
    
    # Set date if not provided
    if date is None:
        date = datetime.now().date()
    elif isinstance(date, str):
        date = datetime.fromisoformat(date.replace('Z', '+00:00')).date()
    elif isinstance(date, datetime):
        date = date.date()
    
    # Platform-specific season effects
    platform_seasons = {
        "facebook": {
            # Months with higher engagement
            "high_engagement_months": [1, 11, 12],  # Jan, Nov, Dec
            # Months with lower CPM
            "low_cpm_months": [1, 2, 8],  # Jan, Feb, Aug
        },
        "instagram": {
            "high_engagement_months": [12, 1, 7],  # Dec, Jan, Jul
            "low_cpm_months": [1, 2, 6],  # Jan, Feb, Jun
        },
        "google": {
            "high_engagement_months": [11, 12, 1],  # Nov, Dec, Jan
            "low_cpm_months": [1, 2, 7],  # Jan, Feb, Jul
        },
        "tiktok": {
            "high_engagement_months": [12, 1, 6],  # Dec, Jan, Jun
            "low_cpm_months": [1, 2, 4],  # Jan, Feb, Apr
        },
        "linkedin": {
            "high_engagement_months": [1, 2, 9],  # Jan, Feb, Sep
            "low_cpm_months": [7, 8, 12],  # Jul, Aug, Dec
        }
    }
    
    # Get platform season data
    platform_data = platform_seasons.get(platform.lower(), {})
    
    current_month = date.month
    
    # Check if current month is a high engagement month for this platform
    high_engagement_months = platform_data.get("high_engagement_months", [])
    features[f"{platform}_high_engagement_month"] = int(current_month in high_engagement_months)
    
    # Check if current month is a low CPM month for this platform
    low_cpm_months = platform_data.get("low_cpm_months", [])
    features[f"{platform}_low_cpm_month"] = int(current_month in low_cpm_months)
    
    # Platform algorithm changes
    # (Simplified example - in practice, you would maintain a database of known algorithm changes)
    algorithm_changes = {
        "facebook": [
            datetime(2021, 1, 15).date(),  # Privacy changes
            datetime(2022, 3, 10).date(),  # News feed algorithm update
            datetime(2023, 2, 22).date(),  # AI-powered ranking
        ],
        "google": [
            datetime(2021, 6, 18).date(),  # Core algorithm update
            datetime(2022, 5, 25).date(),  # Performance Max rollout
            datetime(2023, 4, 10).date(),  # AI bidding enhancements
        ]
    }
    
    # Get platform algorithm changes
    platform_changes = algorithm_changes.get(platform.lower(), [])
    
    # Find the most recent change before current date
    most_recent_change = None
    for change_date in sorted(platform_changes, reverse=True):
        if change_date <= date:
            most_recent_change = change_date
            break
    
    if most_recent_change:
        days_since_change = (date - most_recent_change).days
        features[f"{platform}_days_since_algorithm_change"] = days_since_change
        
        # Recent algorithm change buckets
        change_buckets = [30, 90, 180]
        for bucket in change_buckets:
            features[f"{platform}_algorithm_change_within_{bucket}d"] = int(days_since_change <= bucket)
    
    # Platform usage patterns
    platform_usage = {
        "facebook": {
            "peak_days": [0, 6],  # Monday and Sunday
            "peak_hours": [20, 21, 22]  # 8-10 PM
        },
        "instagram": {
            "peak_days": [1, 3],  # Tuesday and Thursday
            "peak_hours": [12, 17, 21]  # 12 PM, 5 PM, 9 PM
        },
        "linkedin": {
            "peak_days": [2, 3],  # Wednesday and Thursday
            "peak_hours": [9, 12, 17]  # 9 AM, 12 PM, 5 PM
        }
    }
    
    # Get platform usage data
    platform_usage_data = platform_usage.get(platform.lower(), {})
    
    # Check if current date is a peak day for this platform
    if isinstance(date, datetime):
        current_day = date.weekday()
        current_hour = date.hour
    else:
        # If date is just a date object, we can't check hour
        current_day = date.weekday()
        current_hour = None
    
    peak_days = platform_usage_data.get("peak_days", [])
    features[f"{platform}_peak_usage_day"] = int(current_day in peak_days)
    
    # Check if current hour is a peak hour for this platform
    if current_hour is not None:
        peak_hours = platform_usage_data.get("peak_hours", [])
        features[f"{platform}_peak_usage_hour"] = int(current_hour in peak_hours)
    
    return features
``` 

## Audience Features

Audience features capture the characteristics and behaviors of target audiences for advertising campaigns.

### Demographic Features

Features related to audience demographics:

- Age group distributions
- Gender distributions
- Income level distributions
- Education level distributions
- Occupation categories
- Household composition
- Geographic region characteristics

### Interest and Behavior Features

Features related to audience interests and behaviors:

- Interest categories and strengths
- Online behavior patterns
- Purchase behavior profiles
- Device usage patterns
- Content consumption habits
- Brand affinity signals
- Life event indicators

### Audience Targeting Overlap

Features that capture targeting strategy effectiveness:

- Target audience size
- Audience segment overlaps
- Lookalike audience similarity scores
- Custom audience match rates
- Audience reach saturation
- Audience engagement consistency

### Implementation Example

Here's an example of extracting audience demographic features:

```python
def extract_audience_demographic_features(audience_data):
    """Extract demographic features from audience data."""
    features = {}
    
    # Age distribution
    age_distribution = audience_data.get("age_distribution", {})
    age_groups = ["13-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    
    # Process each age group
    for age_group in age_groups:
        # Use normalized distribution (percentages)
        features[f"audience_age_{age_group.replace('-', '_')}"] = age_distribution.get(age_group, 0)
    
    # Calculate audience age metrics
    weighted_sum = 0
    total_weight = 0
    
    # Age group midpoints for weighted calculations
    age_midpoints = {
        "13-17": 15,
        "18-24": 21,
        "25-34": 29.5,
        "35-44": 39.5,
        "45-54": 49.5,
        "55-64": 59.5,
        "65+": 70  # Approximate
    }
    
    for age_group, percentage in age_distribution.items():
        if age_group in age_midpoints:
            weighted_sum += age_midpoints[age_group] * percentage
            total_weight += percentage
    
    # Calculate estimated average age
    if total_weight > 0:
        features["audience_avg_age"] = weighted_sum / total_weight
    else:
        features["audience_avg_age"] = 0
    
    # Age diversity (higher = more diverse)
    if len(age_distribution) > 0:
        features["audience_age_diversity"] = 1 - max(age_distribution.values()) / sum(age_distribution.values())
    else:
        features["audience_age_diversity"] = 0
    
    # Gender distribution
    gender_distribution = audience_data.get("gender_distribution", {})
    
    features["audience_male_pct"] = gender_distribution.get("male", 0)
    features["audience_female_pct"] = gender_distribution.get("female", 0)
    features["audience_other_gender_pct"] = gender_distribution.get("other", 0)
    
    # Gender diversity (higher = more balanced)
    gender_values = [v for v in gender_distribution.values() if v > 0]
    if gender_values:
        features["audience_gender_diversity"] = 1 - max(gender_values) / sum(gender_values)
    else:
        features["audience_gender_diversity"] = 0
    
    # Income distribution
    income_distribution = audience_data.get("income_distribution", {})
    income_brackets = ["under_25k", "25k_50k", "50k_75k", "75k_100k", "100k_150k", "over_150k"]
    
    # Process each income bracket
    for bracket in income_brackets:
        features[f"audience_income_{bracket}"] = income_distribution.get(bracket, 0)
    
    # Estimate average income
    income_midpoints = {
        "under_25k": 12500,
        "25k_50k": 37500,
        "50k_75k": 62500,
        "75k_100k": 87500,
        "100k_150k": 125000,
        "over_150k": 200000  # Approximate
    }
    
    weighted_sum = 0
    total_weight = 0
    
    for bracket, percentage in income_distribution.items():
        if bracket in income_midpoints:
            weighted_sum += income_midpoints[bracket] * percentage
            total_weight += percentage
    
    # Calculate estimated average income
    if total_weight > 0:
        features["audience_avg_income"] = weighted_sum / total_weight
    else:
        features["audience_avg_income"] = 0
    
    # Education distribution
    education_distribution = audience_data.get("education_distribution", {})
    education_levels = ["high_school", "some_college", "associates", "bachelors", "masters", "doctorate"]
    
    # Process each education level
    for level in education_levels:
        features[f"audience_education_{level}"] = education_distribution.get(level, 0)
    
    # Geographic distribution
    geo_distribution = audience_data.get("geo_distribution", {})
    
    # Urban/suburban/rural breakdown
    features["audience_urban_pct"] = geo_distribution.get("urban", 0)
    features["audience_suburban_pct"] = geo_distribution.get("suburban", 0)
    features["audience_rural_pct"] = geo_distribution.get("rural", 0)
    
    # Region distribution
    region_distribution = geo_distribution.get("regions", {})
    major_regions = ["northeast", "southeast", "midwest", "southwest", "west"]
    
    for region in major_regions:
        features[f"audience_region_{region}"] = region_distribution.get(region, 0)
    
    return features
```

### Interest and Behavior Feature Extraction

Extracting audience interest and behavior features:

```python
def extract_audience_interest_features(audience_data):
    """Extract interest and behavior features from audience data."""
    features = {}
    
    # Interest categories
    interest_distribution = audience_data.get("interest_distribution", {})
    
    # Major interest categories
    interest_categories = [
        "technology", "fashion", "sports", "travel", "food", 
        "fitness", "entertainment", "business", "education",
        "home", "beauty", "automotive", "finance", "shopping"
    ]
    
    # Process each interest category
    for category in interest_categories:
        features[f"audience_interest_{category}"] = interest_distribution.get(category, 0)
    
    # Interest diversity (higher = more diverse interests)
    interest_values = [v for v in interest_distribution.values() if v > 0]
    if interest_values:
        features["audience_interest_diversity"] = 1 - max(interest_values) / sum(interest_values)
    else:
        features["audience_interest_diversity"] = 0
    
    # Count number of significant interests (>10%)
    features["audience_significant_interest_count"] = sum(1 for v in interest_distribution.values() if v > 0.1)
    
    # Purchase behavior
    purchase_behavior = audience_data.get("purchase_behavior", {})
    
    # Purchase frequency
    frequency_distribution = purchase_behavior.get("frequency", {})
    purchase_frequencies = ["frequent", "average", "infrequent", "rare"]
    
    for freq in purchase_frequencies:
        features[f"audience_purchase_{freq}"] = frequency_distribution.get(freq, 0)
    
    # Purchase categories
    purchase_categories = purchase_behavior.get("categories", {})
    major_purchase_categories = [
        "apparel", "electronics", "groceries", "home_goods", 
        "beauty", "health", "entertainment", "travel"
    ]
    
    for category in major_purchase_categories:
        features[f"audience_purchases_{category}"] = purchase_categories.get(category, 0)
    
    # Price sensitivity
    price_sensitivity = purchase_behavior.get("price_sensitivity", {})
    sensitivity_levels = ["low", "medium", "high"]
    
    for level in sensitivity_levels:
        features[f"audience_price_sensitivity_{level}"] = price_sensitivity.get(level, 0)
    
    # Device usage
    device_usage = audience_data.get("device_usage", {})
    devices = ["smartphone", "tablet", "desktop", "connected_tv"]
    
    for device in devices:
        features[f"audience_device_{device}"] = device_usage.get(device, 0)
    
    # Primary device (highest usage)
    if device_usage:
        primary_device = max(device_usage.items(), key=lambda x: x[1])[0]
        for device in devices:
            features[f"audience_primary_device_{device}"] = int(device == primary_device)
    
    # Platform usage
    platform_usage = audience_data.get("platform_usage", {})
    platforms = ["facebook", "instagram", "twitter", "youtube", "tiktok", "linkedin", "pinterest"]
    
    for platform in platforms:
        features[f"audience_platform_{platform}"] = platform_usage.get(platform, 0)
    
    # Life events
    life_events = audience_data.get("life_events", {})
    events = ["recent_move", "new_job", "engagement", "marriage", "new_child", "retirement"]
    
    for event in events:
        features[f"audience_life_event_{event}"] = life_events.get(event, 0)
    
    return features
```

### Audience Targeting Features

Extracting audience targeting effectiveness features:

```python
def extract_audience_targeting_features(audience_data, campaign_data):
    """Extract features related to audience targeting effectiveness."""
    features = {}
    
    # Audience size metrics
    audience_size = audience_data.get("audience_size", 0)
    features["audience_size"] = audience_size
    
    # Size brackets
    size_brackets = [10000, 50000, 100000, 500000, 1000000, 5000000]
    for i, threshold in enumerate(size_brackets):
        if i == 0:
            features[f"audience_size_under_{threshold}"] = int(audience_size < threshold)
        else:
            prev_threshold = size_brackets[i-1]
            features[f"audience_size_{prev_threshold}_to_{threshold}"] = int(
                prev_threshold <= audience_size < threshold
            )
    
    features[f"audience_size_over_{size_brackets[-1]}"] = int(
        audience_size >= size_brackets[-1]
    )
    
    # Audience reach
    potential_reach = audience_data.get("potential_reach", 0)
    actual_reach = campaign_data.get("reach", 0)
    
    if potential_reach > 0:
        reach_pct = min(100, max(0, (actual_reach / potential_reach) * 100))
        features["audience_reach_pct"] = reach_pct
        
        # Reach brackets
        reach_brackets = [20, 40, 60, 80]
        for i, threshold in enumerate(reach_brackets):
            if i == 0:
                features[f"audience_reach_under_{threshold}pct"] = int(reach_pct < threshold)
            else:
                prev_threshold = reach_brackets[i-1]
                features[f"audience_reach_{prev_threshold}pct_to_{threshold}pct"] = int(
                    prev_threshold <= reach_pct < threshold
                )
        
        features[f"audience_reach_over_{reach_brackets[-1]}pct"] = int(
            reach_pct >= reach_brackets[-1]
        )
    
    # Audience match rates
    if "custom_audiences" in audience_data:
        custom_audiences = audience_data["custom_audiences"]
        
        if custom_audiences:
            # Average match rate across custom audiences
            match_rates = [aud.get("match_rate", 0) for aud in custom_audiences if "match_rate" in aud]
            if match_rates:
                features["audience_avg_match_rate"] = sum(match_rates) / len(match_rates)
                features["audience_min_match_rate"] = min(match_rates)
                features["audience_max_match_rate"] = max(match_rates)
    
    # Lookalike audiences
    if "lookalike_audiences" in audience_data:
        lookalike_audiences = audience_data["lookalike_audiences"]
        
        if lookalike_audiences:
            # Average similarity across lookalike audiences
            similarities = [aud.get("similarity", 0) for aud in lookalike_audiences if "similarity" in aud]
            if similarities:
                features["audience_avg_lookalike_similarity"] = sum(similarities) / len(similarities)
                features["audience_min_lookalike_similarity"] = min(similarities)
                features["audience_max_lookalike_similarity"] = max(similarities)
    
    # Audience fragmentation
    if "segments" in audience_data:
        segments = audience_data["segments"]
        
        # Number of segments
        features["audience_segment_count"] = len(segments)
        
        # Calculate segment entropy (higher = more evenly distributed)
        if segments:
            segment_sizes = [seg.get("size", 0) for seg in segments]
            total_size = sum(segment_sizes)
            
            if total_size > 0:
                import math
                entropy = 0
                for size in segment_sizes:
                    prob = size / total_size
                    if prob > 0:
                        entropy -= prob * math.log2(prob)
                
                features["audience_segment_entropy"] = entropy
    
    # Audience relevance score (if available)
    features["audience_relevance_score"] = audience_data.get("relevance_score", 0)
    
    return features
``` 

## Performance History Features

Performance history features capture patterns from past campaign performance that can predict future results.

### Historical Performance Metrics

Basic historical performance metrics:

- Historical click-through rates (CTR)
- Historical conversion rates
- Historical cost per click (CPC)
- Historical cost per acquisition (CPA)
- Historical return on ad spend (ROAS)
- Historical engagement rates
- Historical quality scores

### Trend Features

Features capturing performance trends over time:

- Day-over-day changes
- Week-over-week changes
- Month-over-month changes
- Moving averages (7-day, 14-day, 30-day)
- Growth/decline rates
- Volatility measures
- Seasonally adjusted trends

### Performance Distribution Features

Features capturing performance variation:

- Performance percentiles
- Performance standard deviations
- Performance coefficient of variation
- Performance outlier frequencies
- Performance range (min-max spread)
- Performance consistency measures

### Implementation Example

Extracting basic historical performance features:

```python
def extract_historical_performance_features(performance_history):
    """Extract features from historical performance data."""
    features = {}
    
    # Basic metrics
    metrics = ["impressions", "clicks", "conversions", "spend", "revenue"]
    
    # Time windows (in days)
    windows = [7, 14, 30, 90]
    
    # For each time window
    for window in windows:
        window_data = performance_history.get(f"last_{window}d", {})
        
        # Skip if no data for this window
        if not window_data:
            continue
        
        # Extract basic metrics
        for metric in metrics:
            if metric in window_data:
                features[f"hist_{window}d_{metric}"] = window_data[metric]
        
        # Calculate derived metrics
        if "impressions" in window_data and window_data["impressions"] > 0:
            # CTR
            if "clicks" in window_data:
                features[f"hist_{window}d_ctr"] = window_data["clicks"] / window_data["impressions"]
            
            # CPM
            if "spend" in window_data:
                features[f"hist_{window}d_cpm"] = (window_data["spend"] / window_data["impressions"]) * 1000
        
        if "clicks" in window_data and window_data["clicks"] > 0:
            # CPC
            if "spend" in window_data:
                features[f"hist_{window}d_cpc"] = window_data["spend"] / window_data["clicks"]
            
            # Conversion Rate
            if "conversions" in window_data:
                features[f"hist_{window}d_conv_rate"] = window_data["conversions"] / window_data["clicks"]
        
        if "conversions" in window_data and window_data["conversions"] > 0:
            # CPA
            if "spend" in window_data:
                features[f"hist_{window}d_cpa"] = window_data["spend"] / window_data["conversions"]
        
        if "spend" in window_data and window_data["spend"] > 0:
            # ROAS
            if "revenue" in window_data:
                features[f"hist_{window}d_roas"] = window_data["revenue"] / window_data["spend"]
    
    return features
```

### Trend Feature Extraction

Extracting trend-based features from historical data:

```python
import numpy as np
from scipy import stats

def extract_trend_features(performance_history, metrics=None):
    """Extract trend features from historical performance data."""
    features = {}
    
    # Default metrics to analyze
    if metrics is None:
        metrics = ["impressions", "clicks", "ctr", "conversions", "conv_rate", "cpa", "roas"]
    
    # Get daily data (last 30 days)
    daily_data = performance_history.get("daily_data", [])
    
    # Skip if not enough data
    if len(daily_data) < 7:
        return features
    
    # Sort by date
    daily_data = sorted(daily_data, key=lambda x: x.get("date", ""))
    
    # Get last 7, 14, and 30 days of data
    last_7_days = daily_data[-7:]
    last_14_days = daily_data[-14:] if len(daily_data) >= 14 else daily_data
    last_30_days = daily_data[-30:] if len(daily_data) >= 30 else daily_data
    
    # For each metric
    for metric in metrics:
        # Skip metrics not in the data
        if not all(metric in day for day in last_7_days):
            continue
        
        # Calculate week-over-week change
        if len(daily_data) >= 14:
            current_week = sum(day[metric] for day in last_7_days)
            previous_week = sum(day[metric] for day in daily_data[-14:-7])
            
            if previous_week > 0:
                wow_change = (current_week - previous_week) / previous_week
                features[f"{metric}_wow_change"] = wow_change
                
                # WoW change brackets
                wow_brackets = [-0.5, -0.2, 0, 0.2, 0.5]
                for i, threshold in enumerate(wow_brackets):
                    if i == 0:
                        features[f"{metric}_wow_under_{abs(threshold):.1f}"] = int(wow_change < threshold)
                    else:
                        prev_threshold = wow_brackets[i-1]
                        features[f"{metric}_wow_{prev_threshold:.1f}_to_{threshold:.1f}"] = int(
                            prev_threshold <= wow_change < threshold
                        )
                
                features[f"{metric}_wow_over_{wow_brackets[-1]:.1f}"] = int(
                    wow_change >= wow_brackets[-1]
                )
        
        # Calculate moving averages
        values_7d = [day[metric] for day in last_7_days]
        values_14d = [day[metric] for day in last_14_days]
        values_30d = [day[metric] for day in last_30_days]
        
        features[f"{metric}_7d_avg"] = np.mean(values_7d)
        features[f"{metric}_14d_avg"] = np.mean(values_14d)
        features[f"{metric}_30d_avg"] = np.mean(values_30d)
        
        # Calculate standard deviations
        features[f"{metric}_7d_std"] = np.std(values_7d)
        features[f"{metric}_14d_std"] = np.std(values_14d)
        features[f"{metric}_30d_std"] = np.std(values_30d)
        
        # Calculate coefficient of variation (CV)
        if features[f"{metric}_7d_avg"] > 0:
            features[f"{metric}_7d_cv"] = features[f"{metric}_7d_std"] / features[f"{metric}_7d_avg"]
        
        if features[f"{metric}_30d_avg"] > 0:
            features[f"{metric}_30d_cv"] = features[f"{metric}_30d_std"] / features[f"{metric}_30d_avg"]
        
        # Calculate trend (linear regression slope)
        # A positive slope indicates an upward trend, negative indicates downward
        x_values = np.array(range(len(values_30d)))
        y_values = np.array(values_30d)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
        
        features[f"{metric}_30d_trend_slope"] = slope
        features[f"{metric}_30d_trend_r_value"] = r_value
        features[f"{metric}_30d_trend_p_value"] = p_value
        
        # Trend strength and direction
        features[f"{metric}_trend_strength"] = abs(r_value)
        features[f"{metric}_trend_direction"] = 1 if slope > 0 else (-1 if slope < 0 else 0)
        
        # Check for trend significance
        features[f"{metric}_significant_trend"] = int(p_value < 0.05)
        
        # Detect outliers (Z-score > 2)
        z_scores = np.abs(stats.zscore(values_30d))
        features[f"{metric}_outlier_count"] = sum(z > 2 for z in z_scores)
        features[f"{metric}_has_outliers"] = int(features[f"{metric}_outlier_count"] > 0)
        
        # Calculate performance stability
        # Higher value = more stable performance
        if len(values_30d) > 1:
            features[f"{metric}_stability"] = 1 / (1 + features[f"{metric}_30d_cv"])
    
    return features
```

### Performance Comparison Features

Extracting features that compare performance to benchmarks:

```python
def extract_performance_comparison_features(performance_data, benchmark_data):
    """Extract features comparing performance to benchmarks."""
    features = {}
    
    # Skip if no benchmark data
    if not benchmark_data:
        return features
    
    # Metrics to compare
    metrics = ["ctr", "conv_rate", "cpc", "cpa", "roas"]
    
    # For each metric
    for metric in metrics:
        # Skip metrics not in the data
        if metric not in performance_data or metric not in benchmark_data:
            continue
        
        performance_value = performance_data[metric]
        benchmark_value = benchmark_data[metric]
        
        # Skip if benchmark value is zero
        if benchmark_value == 0:
            continue
        
        # Calculate relative performance
        relative_performance = performance_value / benchmark_value
        features[f"{metric}_vs_benchmark"] = relative_performance
        
        # Performance brackets
        if metric in ["ctr", "conv_rate", "roas"]:
            # For metrics where higher is better
            if relative_performance < 0.5:
                performance_bracket = "very_poor"
            elif relative_performance < 0.8:
                performance_bracket = "poor"
            elif relative_performance < 1.2:
                performance_bracket = "average"
            elif relative_performance < 2.0:
                performance_bracket = "good"
            else:
                performance_bracket = "excellent"
        else:
            # For metrics where lower is better (cpc, cpa)
            if relative_performance > 2.0:
                performance_bracket = "very_poor"
            elif relative_performance > 1.2:
                performance_bracket = "poor"
            elif relative_performance > 0.8:
                performance_bracket = "average"
            elif relative_performance > 0.5:
                performance_bracket = "good"
            else:
                performance_bracket = "excellent"
        
        features[f"{metric}_performance_bracket"] = performance_bracket
        
        # One-hot encode performance bracket
        brackets = ["very_poor", "poor", "average", "good", "excellent"]
        for bracket in brackets:
            features[f"{metric}_perf_{bracket}"] = int(performance_bracket == bracket)
    
    # Overall performance score (0-100)
    # Higher values indicate better performance
    score_components = []
    
    if "ctr_vs_benchmark" in features:
        ctr_score = min(100, features["ctr_vs_benchmark"] * 50)
        score_components.append(ctr_score)
    
    if "conv_rate_vs_benchmark" in features:
        conv_score = min(100, features["conv_rate_vs_benchmark"] * 50)
        score_components.append(conv_score)
    
    if "cpc_vs_benchmark" in features:
        # Lower CPC is better, so invert the ratio
        cpc_score = min(100, (1 / features["cpc_vs_benchmark"]) * 50)
        score_components.append(cpc_score)
    
    if "roas_vs_benchmark" in features:
        roas_score = min(100, features["roas_vs_benchmark"] * 50)
        score_components.append(roas_score)
    
    if score_components:
        features["overall_performance_score"] = sum(score_components) / len(score_components)
    
    return features
``` 

## Feature Selection

Feature selection is critical for building efficient, interpretable, and high-performing models. This section describes our approach to selecting the most relevant features.

### Feature Selection Approach

Our feature selection process follows these steps:

1. **Initial Feature Pool**: Generate a comprehensive set of candidate features
2. **Feature Filtering**: Remove redundant, irrelevant, or noisy features
3. **Feature Ranking**: Prioritize features based on importance scores
4. **Feature Subset Selection**: Select the optimal subset of features
5. **Validation**: Verify performance impact of selected features

### Feature Filtering Methods

Methods used to filter out less useful features:

- **Variance Thresholding**: Remove features with low variance
- **Correlation Analysis**: Remove highly correlated features
- **Missing Value Analysis**: Remove features with excessive missing values
- **Feature Stability Analysis**: Remove unstable features across data subsets

### Feature Ranking Methods

Methods used to rank feature importance:

- **Statistical Tests**: Chi-squared, ANOVA, mutual information
- **Model-based Importance**: Tree-based feature importance
- **Permutation Importance**: Measuring impact by feature permutation
- **SHAP Values**: Using Shapley values to quantify feature contributions

### Feature Subset Selection

Approaches for selecting the optimal feature subset:

- **Wrapper Methods**: Recursive feature elimination
- **Embedded Methods**: L1 regularization (Lasso)
- **Ensemble Selection**: Stability selection across multiple models
- **Domain Knowledge**: Expert-guided feature selection

### Implementation Example

Here's an example of implementing a feature selection pipeline:

```python
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

def select_features(X, y, n_features=None, method='combined'):
    """
    Select the most important features.
    
    Args:
        X: Feature matrix
        y: Target variable
        n_features: Number of features to select (default: auto-determined)
        method: Feature selection method ('variance', 'mi', 'rf', 'combined')
        
    Returns:
        selected_features: List of selected feature names
        importance_scores: Dictionary of feature importance scores
    """
    # Get feature names
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_values = X.values
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_values = X
    
    # Initialize importance dictionary
    importance_scores = {feature: 0 for feature in feature_names}
    
    # Step 1: Remove low-variance features
    if method in ['variance', 'combined']:
        print("Applying variance thresholding...")
        
        # Scale features for fair comparison
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_values)
        
        # Apply variance threshold
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(X_scaled)
        
        # Get selected features
        support = selector.get_support()
        
        # Update importance for removed features
        for i, selected in enumerate(support):
            if not selected:
                importance_scores[feature_names[i]] = 0
                
        # Filter features
        X_values = selector.transform(X_values)
        remaining_features = [f for i, f in enumerate(feature_names) if support[i]]
        
        print(f"Removed {len(feature_names) - len(remaining_features)} low-variance features.")
        feature_names = remaining_features
    
    # Step 2: Apply mutual information
    if method in ['mi', 'combined']:
        print("Calculating mutual information...")
        
        # Calculate mutual information
        mi_selector = SelectKBest(mutual_info_regression, k='all')
        mi_selector.fit(X_values, y)
        
        # Get scores
        mi_scores = mi_selector.scores_
        
        # Update importance scores
        for i, feature in enumerate(feature_names):
            importance_scores[feature] += mi_scores[i]
    
    # Step 3: Use random forest feature importance
    if method in ['rf', 'combined']:
        print("Calculating random forest feature importance...")
        
        # Train random forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_values, y)
        
        # Get feature importance
        rf_importance = rf.feature_importances_
        
        # Update importance scores
        for i, feature in enumerate(feature_names):
            importance_scores[feature] += rf_importance[i]
    
    # Step 4: Calculate permutation importance
    if method in ['permutation', 'combined']:
        print("Calculating permutation importance...")
        
        # Train a simple model
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_values, y)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            rf, X_values, y, n_repeats=10, random_state=42
        )
        
        # Update importance scores
        for i, feature in enumerate(feature_names):
            importance_scores[feature] += perm_importance.importances_mean[i]
    
    # Normalize importance scores to 0-1 range
    max_importance = max(importance_scores.values())
    if max_importance > 0:
        for feature in importance_scores:
            importance_scores[feature] /= max_importance
    
    # Sort features by importance
    sorted_features = sorted(
        importance_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Determine number of features to select
    if n_features is None:
        # Use elbow method
        importances = [score for _, score in sorted_features]
        n_features = find_elbow_point(importances) + 1
    
    # Select top features
    selected_features = [feature for feature, _ in sorted_features[:n_features]]
    
    print(f"Selected {len(selected_features)} features.")
    
    return selected_features, importance_scores

def find_elbow_point(values):
    """Find the elbow point in a curve using the maximum curvature method."""
    values = np.array(values)
    n_points = len(values)
    
    # Create x values (indices)
    x = np.array(range(n_points))
    
    # Normalize x and y to 0-1 range
    x_norm = x / np.max(x)
    y_norm = values / np.max(values)
    
    # Calculate curvature at each point
    curvature = np.zeros_like(x_norm)
    
    for i in range(1, n_points - 1):
        # Approximate second derivative
        d2y = y_norm[i+1] + y_norm[i-1] - 2 * y_norm[i]
        d2x = x_norm[i+1] + x_norm[i-1] - 2 * x_norm[i]
        
        # First derivative
        dx = (x_norm[i+1] - x_norm[i-1]) / 2
        dy = (y_norm[i+1] - y_norm[i-1]) / 2
        
        # Curvature formula
        numerator = abs(d2y * dx - dy * d2x)
        denominator = (dx**2 + dy**2)**1.5
        
        if denominator > 0:
            curvature[i] = numerator / denominator
    
    # Find index of maximum curvature
    elbow_index = np.argmax(curvature)
    
    return elbow_index
```

### Feature Selection for Different Models

Our feature selection strategy varies by model:

| Model | Selection Approach | Feature Count |
|-------|-------------------|---------------|
| Ad Score Predictor | Combined (RF + MI + Permutation) | 80-120 features |
| Account Health Predictor | Combined with domain expertise | 100-150 features |
| Ad Sentiment Analyzer | Primarily NLP-based with ablation testing | 50-80 features |

## Feature Importance

Understanding feature importance is crucial for model interpretability and performance optimization.

### Feature Importance Methods

We use several methods to quantify feature importance:

- **Tree-based Importance**: From Random Forest and Gradient Boosting models
- **Permutation Importance**: Measuring performance drop when features are permuted
- **SHAP Values**: Game-theoretic approach for additive feature attribution
- **Partial Dependence Plots**: Visualizing feature effects on model output
- **Feature Ablation**: Measuring performance without specific features

### Global vs. Local Importance

Our feature importance analysis operates at two levels:

- **Global Importance**: Overall contribution across the entire dataset
- **Local Importance**: Instance-specific feature contributions

### Ad Score Predictor Feature Importance

Top features in the Ad Score Predictor model:

| Feature | Importance | Category |
|---------|------------|----------|
| sentiment_score | 0.082 | Text |
| emotion_anticipation | 0.064 | Text |
| has_cta | 0.057 | Text |
| benefit_clarity | 0.053 | Text |
| audience_interest_match | 0.049 | Audience |
| historical_ctr | 0.047 | Performance |
| creative_type | 0.043 | Campaign |
| brand_mention | 0.041 | Text |
| urgency_signal | 0.038 | Text |
| price_mention | 0.036 | Text |

### Account Health Predictor Feature Importance

Top features in the Account Health Predictor model:

| Feature | Importance | Category |
|---------|------------|----------|
| performance_trend_30d | 0.089 | Performance |
| roas_stability | 0.071 | Performance |
| relative_cpa | 0.065 | Cross-Platform |
| audience_reach_saturation | 0.058 | Audience |
| campaign_diversity | 0.054 | Campaign |
| creative_freshness | 0.049 | Campaign |
| budget_utilization | 0.047 | Campaign |
| competitive_density | 0.043 | Cross-Platform |
| quality_score_avg | 0.041 | Performance |
| seasonal_alignment | 0.038 | Temporal |

### Implementation Example

Here's an example of calculating and visualizing SHAP values:

```python
import shap
import matplotlib.pyplot as plt
import numpy as np

def calculate_shap_values(model, X, feature_names=None):
    """
    Calculate SHAP values for a model.
    
    Args:
        model: Trained model (scikit-learn compatible)
        X: Feature matrix
        feature_names: List of feature names
        
    Returns:
        shap_values: SHAP values
        shap_expected_value: Expected value
    """
    # Convert to numpy if pandas DataFrame
    if hasattr(X, 'values'):
        if feature_names is None:
            feature_names = X.columns.tolist()
        X = X.values
    
    # If feature names not provided, create generic ones
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Create explainer
    if hasattr(model, 'predict_proba'):
        explainer = shap.Explainer(model)
    else:
        explainer = shap.Explainer(model)
    
    # Calculate SHAP values
    shap_values = explainer(X)
    
    return shap_values, feature_names

def plot_shap_summary(shap_values, feature_names, plot_type='bar', max_features=20):
    """
    Plot SHAP value summary.
    
    Args:
        shap_values: SHAP values from calculate_shap_values
        feature_names: List of feature names
        plot_type: Type of plot ('bar', 'beeswarm', 'violin')
        max_features: Maximum number of features to show
    """
    plt.figure(figsize=(12, max(8, max_features * 0.4)))
    
    if plot_type == 'bar':
        shap.plots.bar(shap_values, max_display=max_features, show=False)
    elif plot_type == 'beeswarm':
        shap.plots.beeswarm(shap_values, max_display=max_features, show=False)
    elif plot_type == 'violin':
        shap.plots.violin(shap_values, max_display=max_features, show=False)
    
    plt.tight_layout()
    plt.savefig(f"shap_{plot_type}_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_shap_dependence(shap_values, X, feature_idx, interaction_idx=None):
    """
    Plot SHAP dependence plot for a feature.
    
    Args:
        shap_values: SHAP values from calculate_shap_values
        X: Feature matrix
        feature_idx: Index of feature to plot
        interaction_idx: Index of interaction feature (optional)
    """
    plt.figure(figsize=(10, 6))
    
    if interaction_idx is not None:
        shap.plots.scatter(shap_values[:, feature_idx], 
                          color=shap_values[:, interaction_idx],
                          show=False)
    else:
        shap.plots.scatter(shap_values[:, feature_idx], show=False)
    
    plt.tight_layout()
    plt.savefig(f"shap_dependence_{feature_idx}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_shap_waterfall(shap_values, instance_idx, max_features=10):
    """
    Plot SHAP waterfall plot for a specific instance.
    
    Args:
        shap_values: SHAP values from calculate_shap_values
        instance_idx: Index of instance to explain
        max_features: Maximum number of features to show
    """
    plt.figure(figsize=(12, 8))
    
    shap.plots.waterfall(shap_values[instance_idx], max_display=max_features, show=False)
    
    plt.tight_layout()
    plt.savefig(f"shap_waterfall_{instance_idx}.png", dpi=300, bbox_inches='tight')
    plt.show()
```

### Feature Importance Visualization

We use several visualization techniques to communicate feature importance:

1. **SHAP Summary Plot**: Shows the distribution of SHAP values for each feature
2. **Feature Importance Bar Chart**: Shows the average magnitude of feature importance
3. **Partial Dependence Plots**: Shows how features affect the predicted outcome
4. **Feature Interaction Heatmap**: Shows the strength of pairwise feature interactions
5. **Local Explanation Waterfall**: Shows feature contributions for individual predictions

## Implementation Details

This section covers the technical implementation of the feature engineering system in the WITHIN platform.

### Technical Stack

The feature engineering pipeline is built on:

- **Python 3.9+**: Core programming language
- **NumPy & Pandas**: Data manipulation and processing
- **scikit-learn**: Feature transformation and selection
- **PyTorch**: Deep learning components
- **Hugging Face Transformers**: NLP model implementations
- **SHAP**: Feature importance calculation
- **SQLAlchemy**: Database interactions
- **Redis**: Feature caching
- **Ray**: Distributed processing
- **FastAPI**: Feature service endpoints

### System Architecture

The feature engineering system follows a layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                       Client Applications                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    Feature Service API                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   Feature Orchestrator                       │
└─┬───────────────────┬──────────────────┬───────────────────┬┘
  │                   │                  │                   │
┌─▼───────────┐  ┌────▼────────┐  ┌──────▼────────┐  ┌───────▼─────┐
│  Text       │  │  Campaign   │  │  Audience     │  │ Performance │
│  Feature    │  │  Feature    │  │  Feature      │  │ Feature     │
│  Processor  │  │  Processor  │  │  Processor    │  │ Processor   │
└─┬───────────┘  └────┬────────┘  └──────┬────────┘  └───────┬─────┘
  │                   │                  │                   │
┌─▼───────────────────▼──────────────────▼───────────────────▼─────┐
│                    Feature Registry & Cache                       │
└─────────────────────────────┬───────────────────────────────────┬┘
                              │                                   │
┌─────────────────────────────▼───────────┐  ┌───────────────────▼─┐
│              Database                   │  │      Monitoring      │
└───────────────────────────────────────┬─┘  └─────────────────────┘
                                        │
┌───────────────────────────────────────▼─────────────────────────┐
│                      Data Sources                                │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Approach

Our feature engineering implementation follows these best practices:

- **Modular Design**: Components are separate and reusable
- **Pipeline Architecture**: Sequential processing with clear boundaries
- **Caching Strategy**: Multi-level caching to improve performance
- **Validation Hooks**: Built-in validation at each processing stage
- **Extensibility**: Easy to add new feature extractors
- **Monitoring**: Comprehensive logging and monitoring
- **Testing**: Unit and integration tests for all components

### Key Implementation Classes

Main classes in the feature engineering system:

```python
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    """Base class for all feature extractors."""
    
    @abstractmethod
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from input data.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Dictionary of extracted features
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names produced by this extractor.
        
        Returns:
            List of feature names
        """
        pass
    
    def validate_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted features.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Validated features dictionary
        """
        return features

class TextFeatureExtractor(FeatureExtractor):
    """Extracts features from ad text."""
    
    def __init__(self, nlp_model_path: str, use_transformer: bool = True):
        """
        Initialize text feature extractor.
        
        Args:
            nlp_model_path: Path to NLP model
            use_transformer: Whether to use transformer model
        """
        self.nlp_model_path = nlp_model_path
        self.use_transformer = use_transformer
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize NLP models."""
        # Implementation details
        pass
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text features."""
        # Extract ad text from data
        ad_text = data.get("ad_text", "")
        
        if not ad_text:
            return {}
        
        # Extract basic text features
        features = extract_basic_text_features(ad_text)
        
        # Extract transformer features if enabled
        if self.use_transformer:
            transformer_features = extract_transformer_features(ad_text)
            features.update(transformer_features)
        
        # Extract ad-specific features
        ad_features = extract_ad_specific_features(ad_text)
        features.update(ad_features)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        # Implementation details
        return []

class FeaturePipeline:
    """Pipeline for feature extraction, transformation, and selection."""
    
    def __init__(self, extractors: List[FeatureExtractor], selector: Optional[Any] = None):
        """
        Initialize feature pipeline.
        
        Args:
            extractors: List of feature extractors
            selector: Feature selector (optional)
        """
        self.extractors = extractors
        self.selector = selector
        self.feature_cache = {}
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data through the pipeline.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Dictionary of processed features
        """
        # Extract features from each extractor
        all_features = {}
        
        for extractor in self.extractors:
            features = extractor.extract(data)
            all_features.update(features)
        
        # Apply feature selection if available
        if self.selector is not None:
            selected_features = {}
            for feature in self.selector.get_selected_features():
                if feature in all_features:
                    selected_features[feature] = all_features[feature]
            return selected_features
        
        return all_features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names in the pipeline."""
        feature_names = []
        
        for extractor in self.extractors:
            feature_names.extend(extractor.get_feature_names())
        
        return feature_names

class FeatureService:
    """Service for feature extraction and management."""
    
    def __init__(self, pipelines: Dict[str, FeaturePipeline]):
        """
        Initialize feature service.
        
        Args:
            pipelines: Dictionary mapping pipeline names to pipelines
        """
        self.pipelines = pipelines
    
    def extract_features(self, data: Dict[str, Any], pipeline_name: str) -> Dict[str, Any]:
        """
        Extract features using specified pipeline.
        
        Args:
            data: Input data dictionary
            pipeline_name: Name of pipeline to use
            
        Returns:
            Dictionary of extracted features
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not found")
        
        pipeline = self.pipelines[pipeline_name]
        return pipeline.process(data)
    
    def get_pipeline_info(self, pipeline_name: str) -> Dict[str, Any]:
        """
        Get information about a pipeline.
        
        Args:
            pipeline_name: Name of pipeline
            
        Returns:
            Dictionary with pipeline information
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not found")
        
        pipeline = self.pipelines[pipeline_name]
        
        return {
            "name": pipeline_name,
            "feature_count": len(pipeline.get_feature_names()),
            "extractors": [type(ext).__name__ for ext in pipeline.extractors],
            "has_selector": pipeline.selector is not None
        }
```

## Feature Pipelines

Feature pipelines orchestrate the end-to-end process of transforming raw data into model-ready features.

### Pipeline Architecture

Our feature pipelines follow a sequential architecture:

1. **Data Acquisition**: Retrieving data from various sources
2. **Preprocessing**: Cleaning and normalizing input data
3. **Feature Extraction**: Generating features from preprocessed data
4. **Feature Transformation**: Scaling, encoding, and normalizing features
5. **Feature Selection**: Selecting the most relevant features
6. **Feature Validation**: Validating feature quality and distributions
7. **Feature Storage**: Storing features for model training and inference

### Pipeline Configuration

Feature pipelines are configured through YAML files:

```yaml
name: ad_score_feature_pipeline
version: "1.2.3"
description: "Feature pipeline for Ad Score Predictor model"

extractors:
  - name: text_features
    type: TextFeatureExtractor
    config:
      nlp_model_path: models/nlp/ad_text_model
      use_transformer: true
      
  - name: campaign_features
    type: CampaignFeatureExtractor
    config:
      include_targeting: true
      include_creative: true
      
  - name: temporal_features
    type: TemporalFeatureExtractor
    config:
      time_windows: [7, 14, 30, 90]
      include_seasonality: true
      
  - name: audience_features
    type: AudienceFeatureExtractor
    config:
      demographic_features: true
      interest_features: true
      
  - name: performance_features
    type: PerformanceFeatureExtractor
    config:
      metrics: ["ctr", "conv_rate", "cpc", "cpa", "roas"]
      include_trends: true

transformers:
  - name: numeric_scaler
    type: StandardScaler
    apply_to:
      - "campaign_features.daily_budget"
      - "performance_features.*"
      
  - name: categorical_encoder
    type: OneHotEncoder
    apply_to:
      - "campaign_features.objective"
      - "campaign_features.bid_strategy"

selectors:
  - name: importance_selector
    type: ImportanceSelector
    config:
      method: "combined"
      n_features: 100

validators:
  - name: range_validator
    type: RangeValidator
    config:
      features:
        - name: "text_features.sentiment_score"
          min: -1.0
          max: 1.0
        - name: "performance_features.hist_30d_ctr"
          min: 0.0
          max: 1.0
          
  - name: distribution_validator
    type: DistributionValidator
    config:
      features:
        - name: "text_features.word_count"
          distribution: "normal"
          params:
            mean: 25
            std: 15
```

### Pipeline Implementations

We implement specialized pipelines for each model:

| Pipeline | Purpose | Key Features |
|----------|---------|--------------|
| AdScoreFeaturePipeline | Extract features for ad scoring | Text features, campaign settings, historical performance |
| AccountHealthFeaturePipeline | Extract features for account health prediction | Performance trends, campaign diversity, audience targeting |
| SentimentAnalysisFeaturePipeline | Extract features for sentiment analysis | Text linguistics, semantic structure, emotional markers |

### Batch Processing

Feature pipelines support efficient batch processing:

```python
def batch_process_features(data_items, pipeline_name, batch_size=100):
    """
    Process features in batches.
    
    Args:
        data_items: List of data items to process
        pipeline_name: Name of pipeline to use
        batch_size: Batch size
        
    Returns:
        List of processed feature dictionaries
    """
    service = FeatureService(load_pipelines())
    results = []
    
    # Process in batches
    for i in range(0, len(data_items), batch_size):
        batch = data_items[i:i+batch_size]
        
        # Process batch
        batch_results = []
        for item in batch:
            features = service.extract_features(item, pipeline_name)
            batch_results.append(features)
        
        results.extend(batch_results)
        
        print(f"Processed batch {i//batch_size + 1}/{(len(data_items) + batch_size - 1)//batch_size}")
    
    return results
```

## Validation and Testing

Rigorous validation and testing ensure the reliability and quality of our feature engineering process.

### Feature Validation Strategies

We employ several validation strategies:

- **Schema Validation**: Ensure features conform to expected schema
- **Range Validation**: Verify features are within expected ranges
- **Distribution Validation**: Check for distribution shifts
- **Null Value Validation**: Monitor missing value patterns
- **Correlation Validation**: Check for unexpected feature correlations
- **Outlier Detection**: Identify and handle outlier values

### Unit Testing

Unit tests for feature extractors:

```python
import unittest
import numpy as np

class TestTextFeatureExtractor(unittest.TestCase):
    """Unit tests for TextFeatureExtractor."""
    
    def setUp(self):
        """Set up test environment."""
        self.extractor = TextFeatureExtractor(
            nlp_model_path="models/test/nlp_model",
            use_transformer=False
        )
        
        # Sample ad texts
        self.sample_texts = {
            "short": "Limited time offer!",
            "medium": "Get 50% off our premium subscription. Sign up today!",
            "long": "Experience the difference with our revolutionary product. " +
                    "Our cutting-edge technology delivers exceptional results " +
                    "that outperform the competition. Try it risk-free today!"
        }
    
    def test_basic_features(self):
        """Test extraction of basic text features."""
        for name, text in self.sample_texts.items():
            features = self.extractor.extract({"ad_text": text})
            
            # Check that essential features are present
            self.assertIn("char_count", features)
            self.assertIn("word_count", features)
            
            # Check feature values
            self.assertEqual(features["char_count"], len(text))
            self.assertGreater(features["word_count"], 0)
    
    def test_sentiment_features(self):
        """Test extraction of sentiment features."""
        # Positive text
        positive_text = "Amazing product! Love the results!"
        pos_features = self.extractor.extract({"ad_text": positive_text})
        
        # Negative text
        negative_text = "Disappointed with the quality. Wouldn't recommend."
        neg_features = self.extractor.extract({"ad_text": negative_text})
        
        # Check sentiment scores
        self.assertIn("sentiment_score", pos_features)
        self.assertIn("sentiment_score", neg_features)
        
        # Positive text should have higher sentiment score
        self.assertGreater(pos_features["sentiment_score"], 
                          neg_features["sentiment_score"])
    
    def test_edge_cases(self):
        """Test handling of edge cases."""
        # Empty text
        empty_features = self.extractor.extract({"ad_text": ""})
        self.assertEqual(empty_features, {})
        
        # Very long text
        very_long_text = "word " * 1000
        long_features = self.extractor.extract({"ad_text": very_long_text})
        self.assertIn("char_count", long_features)
        
        # Text with special characters
        special_text = "50% OFF! 💥 #amazing deal $99.99 😍"
        special_features = self.extractor.extract({"ad_text": special_text})
        self.assertIn("emoji_count", special_features)
        self.assertGreater(special_features["emoji_count"], 0)
```

### Integration Testing

Integration tests for feature pipelines:

```python
import unittest
import json
import os

class TestFeaturePipeline(unittest.TestCase):
    """Integration tests for feature pipelines."""
    
    def setUp(self):
        """Set up test environment."""
        # Load test data
        with open("tests/data/sample_ads.json", "r") as f:
            self.sample_ads = json.load(f)
        
        # Initialize service
        self.service = FeatureService(load_test_pipelines())
    
    def test_ad_score_pipeline(self):
        """Test ad score feature pipeline."""
        # Process sample ad
        ad_data = self.sample_ads[0]
        features = self.service.extract_features(ad_data, "ad_score_feature_pipeline")
        
        # Check essential feature categories
        self.assertIn("sentiment_score", features)
        self.assertIn("word_count", features)
        self.assertIn("has_cta", features)
        
        # Check campaign features
        self.assertIn("daily_budget", features)
        self.assertIn("objective_encoded", features)
        
        # Check expected feature count
        self.assertGreaterEqual(len(features), 50)
    
    def test_batch_processing(self):
        """Test batch processing of features."""
        # Process batch of ads
        batch_results = batch_process_features(
            self.sample_ads[:5], 
            "ad_score_feature_pipeline",
            batch_size=2
        )
        
        # Check results
        self.assertEqual(len(batch_results), 5)
        for result in batch_results:
            self.assertGreaterEqual(len(result), 50)
    
    def test_feature_validation(self):
        """Test feature validation."""
        # Create validator
        validator = FeatureValidator(load_validation_rules("ad_score_feature_pipeline"))
        
        # Process sample ad
        ad_data = self.sample_ads[0]
        features = self.service.extract_features(ad_data, "ad_score_feature_pipeline")
        
        # Validate features
        validation_result = validator.validate(features)
        
        # Check validation result
        self.assertTrue(validation_result.is_valid)
        self.assertEqual(len(validation_result.errors), 0)
```

### Performance Testing

We regularly benchmark the performance of our feature engineering system:

- **Throughput**: Features processed per second
- **Latency**: Time to extract features for a single instance
- **Resource Usage**: CPU, memory, and disk utilization
- **Scaling Behavior**: Performance under varying load conditions

### Monitoring in Production

In production, we monitor:

- **Feature Distributions**: Track feature distributions over time
- **Missing Values**: Monitor missing value rates
- **Feature Correlations**: Track changes in feature relationships
- **Resource Usage**: Monitor computation and storage requirements
- **Extraction Errors**: Track and alert on feature extraction errors

## Common Issues and Solutions

This section addresses common challenges in feature engineering for ad performance prediction.

### Data Quality Issues

| Issue | Solution |
|-------|----------|
| Missing campaign data | Implement fallback features and imputation strategies |
| Inconsistent text formatting | Apply robust text preprocessing pipelines |
| Platform-specific data fields | Create unified schema with platform-specific adapters |
| Historical data gaps | Use time-aware imputation and interpolation methods |
| Outlier values | Apply robust scaling and winsorization techniques |

### Performance Issues

| Issue | Solution |
|-------|----------|
| Slow text feature extraction | Cache processed text features and use optimized NLP models |
| High memory usage for transformers | Use quantized models and batch processing |
| Feature explosion from one-hot encoding | Apply feature hashing or limit cardinality |
| Slow feature selection | Pre-compute feature importance and cache selections |
| Database bottlenecks | Implement multi-level caching and async loading |

### Feature Drift

| Issue | Solution |
|-------|----------|
| Seasonal feature drift | Incorporate seasonal indicators and adjust for seasonality |
| Platform algorithm changes | Monitor for distribution shifts and retrain models |
| Industry trend shifts | Implement adaptive feature normalization |
| Audience behavior changes | Use relative features rather than absolute metrics |
| Creative format evolution | Design format-agnostic features and regular updates |

### Implementation Challenges

| Issue | Solution |
|-------|----------|
| Feature naming conflicts | Implement consistent feature namespacing |
| Pipeline dependency management | Use explicit dependency declarations and versioning |
| Feature version compatibility | Implement feature schema versioning |
| Reproducibility issues | Use deterministic feature extraction with fixed seeds |
| Cross-platform consistency | Implement unified feature extraction interfaces |

### Troubleshooting Guide

When troubleshooting feature engineering issues:

1. **Validate Input Data**: Ensure input data is present and correctly formatted
2. **Check Feature Distributions**: Look for unexpected distribution shifts
3. **Monitor Extraction Logs**: Check for warnings or errors in extraction
4. **Verify Pipeline Configuration**: Ensure pipeline is properly configured
5. **Test Feature Transformations**: Verify transformations are applied correctly
6. **Inspect Intermediate Results**: Examine outputs at each pipeline stage
7. **Compare with Previous Versions**: Check for changes from previous runs
8. **Review Feature Importance**: Verify important features are present and correct

## Related Documentation

For more detailed information on the WITHIN feature engineering system, see the following resources:

- [NLP Pipeline Implementation](/docs/implementation/ml/nlp_pipeline.md) *(Implemented)*
- [Ad Score Prediction](/docs/implementation/ml/ad_score_prediction.md) *(Planned - Not yet implemented)*
- [Model Training Process](/docs/implementation/ml/model_training.md) *(Planned - Not yet implemented)*
- [Model Evaluation](/docs/implementation/ml/model_evaluation.md) *(Planned - Not yet implemented)*

> **Note**: Some of the linked documents above are currently planned but not yet implemented. Please refer to the [Documentation Tracker](/docs/implementation/documentation_tracker.md) for the current status of all documentation.