# Sentiment Analysis Methodology

**IMPLEMENTATION STATUS: IMPLEMENTED**


This document details the sentiment analysis methodology used in the WITHIN Ad Sentiment Analyzer, covering the technical approach, implementation details, and evaluation methods.

## Table of Contents

- [Overview](#overview)
- [Technical Approach](#technical-approach)
- [Model Architecture](#model-architecture)
- [Training Data](#training-data)
- [Feature Engineering](#feature-engineering)
- [Aspect-Based Sentiment Analysis](#aspect-based-sentiment-analysis)
- [Contextual Adaptation](#contextual-adaptation)
- [Implementation Details](#implementation-details)
- [Evaluation Methodology](#evaluation-methodology)
- [Performance Metrics](#performance-metrics)
- [Integration Guidelines](#integration-guidelines)
- [Limitations and Considerations](#limitations-and-considerations)
- [Future Improvements](#future-improvements)

## Overview

The WITHIN Ad Sentiment Analyzer employs a hybrid approach to sentiment analysis, combining transformer-based deep learning models with lexicon-based methods and rule-based systems. This approach enables accurate sentiment detection specific to advertising content across various platforms and industries.

### Key Features

- Multi-level sentiment analysis (document, sentence, aspect)
- Advertising-specific sentiment lexicon
- Industry and platform contextual adaptation
- Multiple sentiment dimensions (positive/negative, emotional dimensions)
- Confidence scoring for predictions
- Explanations for sentiment assessments

## Technical Approach

The sentiment analysis system uses a three-tier approach to maximize accuracy and performance.

### Hybrid Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Ad Text Input                          │
└───────────────────────────────┬─────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────┐
│                  Text Preprocessing Pipeline                │
└───────────────────────────────┬─────────────────────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
┌─────────────▼──────────────┐   ┌────────────────▼─────────────┐
│  Deep Learning Pipeline    │   │     Lexicon Pipeline         │
│                           │   │                              │
│  ┌─────────────────────┐  │   │  ┌──────────────────────┐    │
│  │ Transformer Encoder │  │   │  │ Advertising Lexicon  │    │
│  └──────────┬──────────┘  │   │  └───────────┬──────────┘    │
│             │             │   │              │               │
│  ┌──────────▼──────────┐  │   │  ┌───────────▼──────────┐    │
│  │ Sentiment Classifier│  │   │  │ Rule-Based Analysis  │    │
│  └──────────┬──────────┘  │   │  └───────────┬──────────┘    │
│             │             │   │              │               │
│  ┌──────────▼──────────┐  │   │  ┌───────────▼──────────┐    │
│  │ Aspect Extraction   │  │   │  │ Context Handling     │    │
│  └──────────┬──────────┘  │   │  └───────────┬──────────┘    │
│             │             │   │              │               │
└─────────────┼─────────────┘   └──────────────┼───────────────┘
              │                                │
              └────────────────┬───────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    Ensemble Integration                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    Context Adaptation                       │
│                                                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Industry Adapter│  │Platform Adapter │  │Audience Adapt│ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                   Calibration Layer                         │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                     Final Sentiment Scores                  │
└────────────────────────────────────────────────────────────┘
```

### Component Breakdown

1. **Deep Learning Pipeline**:
   - Primary sentiment classification
   - Nuanced sentiment detection
   - Aspect identification
   - Context understanding

2. **Lexicon Pipeline**:
   - Advertising-specific terminology
   - Domain-specific sentiment cues
   - Negation and intensifier handling
   - Industry-specific terminology

3. **Ensemble Integration**:
   - Model weighting based on confidence
   - Feature-specific model selection
   - Confidence scoring

4. **Context Adaptation**:
   - Industry-specific calibration
   - Platform-specific normalization
   - Audience targeting adjustments

5. **Calibration Layer**:
   - Confidence calibration
   - Threshold optimization
   - Final score normalization

## Model Architecture

The deep learning component uses a fine-tuned transformer architecture.

### Base Model

- **Architecture**: RoBERTa
- **Size**: Base (125M parameters)
- **Modifications**: 
  - Additional advertising-specific token embeddings
  - Modified attention mechanism for aspect sensitivity
  - Custom classification head

### Model Structure

```
Input Text
    ↓
Tokenization
    ↓
RoBERTa Encoder
    ↓
Contextual Representations
    ↓
    ├── Document-level Pool → Document Sentiment Head
    ├── Sentence-level Pool → Sentence Sentiment Head
    └── Token-level Features → Aspect Extraction Head
                                    ↓
                              Aspect Sentiment 
                                  Heads
```

### Training Methodology

The model was trained using a multi-stage approach:

1. **Pre-training Extension**: Additional pre-training on advertising corpus
2. **Multi-task Fine-tuning**: Simultaneous training on multiple sentiment tasks
3. **Aspect-specific Fine-tuning**: Specialized training for aspect detection
4. **Adversarial Training**: Improving robustness to input variations

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-5 |
| Batch Size | 32 |
| Sequence Length | 256 tokens |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Dropout | 0.1 |
| Warmup Steps | 1000 |
| Training Epochs | 5 |

## Training Data

The sentiment analysis model was trained on diverse advertising data sources.

### Data Composition

- **Total Training Examples**: 845,000
- **Platforms Covered**: Facebook, Google, LinkedIn, Instagram, TikTok, Twitter
- **Industries**: 37 different industry categories
- **Languages**: Primarily English with limited multilingual support
- **Time Range**: 2018-2023

### Data Annotation

Training data was annotated using a multi-stage process:

1. **Initial Annotation**: Professional annotators with advertising expertise
2. **Consistency Checking**: Inter-annotator agreement verification
3. **Edge Case Enrichment**: Additional annotation for difficult cases
4. **Audience Validation**: Validation with target audience panels

### Class Distribution

| Sentiment Level | Positive | Neutral | Negative |
|-----------------|----------|---------|----------|
| Document-level | 58% | 24% | 18% |
| Sentence-level | 52% | 29% | 19% |
| Aspect-level | 45% | 35% | 20% |

### Quality Assurance

- **Inter-annotator Agreement**: Cohen's Kappa of 0.82
- **Annotation Quality Audits**: Regular reviews of annotation quality
- **Bias Detection**: Continuous monitoring for demographic and topic biases

## Feature Engineering

The sentiment analysis system leverages multiple feature types for accurate assessment.

### Linguistic Features

- Part-of-speech distributions
- Dependency parsing features
- Syntactic complexity measures
- Rhetorical structure elements
- Sentence structure patterns

### Semantic Features

- Semantic role labeling
- Named entity recognition
- Emotional content markers
- Persuasive language patterns
- Value proposition indicators

### Advertising-Specific Features

- Call-to-action language
- Offer terminology
- Product description patterns
- Promotional language markers
- Brand sentiment markers
- Urgency and scarcity signals

### Contextual Features

- Industry context markers
- Platform-specific patterns
- Target audience signals
- Seasonal and temporal factors

## Aspect-Based Sentiment Analysis

The system performs fine-grained sentiment analysis on specific aspects of advertisements.

### Aspect Categories

- **Product Features**: Sentiment toward specific product attributes
- **Price/Value**: Sentiment regarding pricing and value propositions
- **Promotion**: Sentiment toward specific offers or deals
- **Brand**: Brand sentiment within the advertisement
- **User Benefits**: Sentiment toward described benefits
- **Call-to-Action**: Sentiment associated with CTA elements
- **Exclusivity**: Sentiment around exclusivity or special access
- **Time Limitations**: Urgency-related sentiment

### Aspect Extraction

Aspects are extracted using a combination of:
- Named entity recognition
- Dependency parsing
- Semantic role labeling
- Domain-specific pattern matching

### Aspect-Level Sentiment Scoring

Each identified aspect receives individual sentiment scores:
- Polarity score (-1.0 to +1.0)
- Intensity score (0.0 to 1.0)
- Confidence score (0.0 to 1.0)

### Implementation Example

```python
def extract_aspect_sentiment(text, aspects):
    """Extract sentiment for specific aspects in ad text."""
    results = {}
    
    # Process text with model
    encodings = tokenizer(text, return_tensors="pt")
    outputs = model(
        input_ids=encodings.input_ids,
        attention_mask=encodings.attention_mask
    )
    
    # Extract aspects
    detected_aspects = aspect_extractor(outputs.last_hidden_state, text)
    
    # Score each requested aspect
    for aspect in aspects:
        if aspect in detected_aspects:
            aspect_encoding = detected_aspects[aspect]
            sentiment = aspect_sentiment_head(aspect_encoding)
            
            results[aspect] = {
                "polarity": sentiment.polarity,
                "intensity": sentiment.intensity,
                "confidence": sentiment.confidence
            }
        else:
            results[aspect] = None
    
    return results
```

## Contextual Adaptation

The sentiment analyzer adapts to different contexts to improve accuracy.

### Industry Adaptation

- Industry-specific terminology handling
- Vertical-specific sentiment calibration
- Industry benchmark normalization

### Platform Adaptation

- Platform-specific language patterns
- Character and format limitations
- Platform audience expectations
- Platform-specific features (hashtags, etc.)

### Audience Adaptation

- Demographic-aware interpretation
- Target audience value alignment
- Psychographic factor consideration

### Implementation Approach

Contextual adaptation uses adapter modules that modify base sentiment scores:

```
Base Sentiment Score
        ↓
Industry Adapter → Platform Adapter → Audience Adapter
        ↓
Adjusted Sentiment Score
```

## Implementation Details

This section covers the technical implementation of the sentiment analysis system.

### Code Architecture

The sentiment analyzer is implemented as a modular Python package with these components:

1. **Preprocessing Module**: Text cleaning and normalization
2. **Model Module**: Deep learning model implementation
3. **Lexicon Module**: Lexicon-based analysis implementation
4. **Aspect Module**: Aspect extraction and scoring
5. **Ensemble Module**: Integration of multiple analysis methods
6. **Adapter Module**: Contextual adaptation components
7. **API Module**: Interface for external systems

### Preprocessing Implementation

```python
def preprocess_ad_text(text, language="en"):
    """Preprocess ad text for sentiment analysis."""
    # Normalize whitespace and case
    text = normalize_text(text)
    
    # Handle special characters
    text = standardize_characters(text)
    
    # Normalize advertising-specific elements
    text = normalize_ad_elements(text)
    
    # Break into sentences for sentence-level analysis
    sentences = split_into_sentences(text, language)
    
    return {
        "normalized_text": text,
        "sentences": sentences,
        "language": language
    }
```

### Inference Pipeline

```python
def analyze_sentiment(text, context=None):
    """Complete sentiment analysis of advertising text."""
    # Set default context if not provided
    context = context or {
        "industry": "general",
        "platform": "unspecified",
        "target_audience": "general"
    }
    
    # Preprocess text
    processed = preprocess_ad_text(text)
    
    # Get transformer model predictions
    dl_predictions = deep_learning_pipeline(processed["normalized_text"])
    
    # Get lexicon-based predictions
    lexicon_predictions = lexicon_pipeline(processed)
    
    # Combine predictions
    combined = ensemble_predictions(dl_predictions, lexicon_predictions)
    
    # Apply contextual adaptation
    adapted = apply_context_adaptation(combined, context)
    
    # Extract aspects and their sentiment
    aspects = extract_aspect_sentiment(processed, adapted)
    
    # Format final results
    return format_sentiment_results(adapted, aspects)
```

### Deployment Architecture

The sentiment analyzer is deployed as:
- Containerized microservice
- Scalable API endpoints
- Batch processing service
- Integrated SDK component

### Performance Optimization

- Model quantization (INT8)
- ONNX Runtime for inference
- Batch processing for efficiency
- Caching for common patterns
- Asynchronous processing pipeline

## Evaluation Methodology

The sentiment analyzer was evaluated using a comprehensive approach.

### Evaluation Datasets

- **Main Test Set**: 25,000 annotated ad examples (held-out)
- **Platform-Specific Sets**: Specialized test sets for each platform
- **Industry-Specific Sets**: Test sets covering major industries
- **Challenging Cases**: Set of edge cases and difficult examples
- **Adversarial Examples**: Deliberately challenging examples

### Cross-Validation

- 5-fold cross-validation during development
- Stratified sampling by platform and industry
- Temporal validation (training on past, testing on recent)

### Human Evaluation

In addition to automated metrics, human evaluators assessed:
- Accuracy of sentiment predictions
- Relevance of identified aspects
- Appropriateness of sentiment intensity
- Usefulness of explanations

### Baseline Comparisons

Performance was compared against:
- General-purpose sentiment analyzers (VADER, TextBlob)
- Commercial sentiment APIs
- Previous version of WITHIN sentiment analyzer
- Human annotator benchmark

## Performance Metrics

The sentiment analyzer's performance varies by context and task.

### Overall Performance

| Metric | Score |
|--------|-------|
| Accuracy | 87.5% |
| Macro F1 | 0.845 |
| Precision | 0.858 |
| Recall | 0.833 |
| ROC AUC | 0.932 |

### Performance by Platform

| Platform | Accuracy | F1 Score |
|----------|----------|----------|
| Facebook | 88.2% | 0.856 |
| Google | 89.1% | 0.872 |
| LinkedIn | 86.5% | 0.845 |
| Instagram | 85.8% | 0.839 |
| TikTok | 83.2% | 0.814 |
| Twitter | 84.7% | 0.827 |

### Performance by Industry

| Industry Category | Accuracy | F1 Score |
|-------------------|----------|----------|
| E-commerce | 89.3% | 0.875 |
| Financial Services | 86.1% | 0.841 |
| Technology | 88.5% | 0.867 |
| Healthcare | 85.2% | 0.835 |
| Travel & Hospitality | 87.6% | 0.857 |
| Entertainment | 86.4% | 0.842 |

### Aspect-Based Performance

| Aspect | Precision | Recall | F1 Score |
|--------|-----------|--------|----------|
| Product Features | 0.843 | 0.817 | 0.830 |
| Price/Value | 0.872 | 0.856 | 0.864 |
| Promotion | 0.891 | 0.879 | 0.885 |
| Brand | 0.865 | 0.842 | 0.853 |
| User Benefits | 0.837 | 0.812 | 0.824 |
| Call-to-Action | 0.883 | 0.866 | 0.874 |

## Integration Guidelines

This section provides guidance for integrating the sentiment analyzer into other systems.

### API Integration

```python
# Example API request
import requests
import json

def analyze_ad_sentiment(ad_text, api_key, context=None):
    """Analyze ad sentiment using the WITHIN API."""
    url = "https://api.within.co/api/v1/analyze/sentiment"
    
    payload = {
        "text": ad_text,
        "context": context or {
            "industry": "general",
            "platform": "facebook",
            "target_audience": "general"
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

### SDK Integration

```python
from within import Client

# Initialize client
client = Client(api_key="your_api_key")

# Analyze sentiment
result = client.analyze_sentiment(
    text="Limited time offer! 50% off our premium subscription plan. Sign up today!",
    context={
        "industry": "software",
        "platform": "facebook",
        "target_audience": "professionals"
    }
)

# Process results
print(f"Overall sentiment: {result.sentiment.overall}")
print(f"Confidence: {result.confidence}")

# Process aspect-level sentiment
for aspect, data in result.aspects.items():
    print(f"Aspect: {aspect}")
    print(f"  Sentiment: {data.sentiment}")
    print(f"  Confidence: {data.confidence}")
```

### Batch Processing

For large-scale analysis, use the batch processing endpoint:

```python
def batch_analyze_sentiment(ads, api_key):
    """Analyze sentiment for multiple ads in batch."""
    url = "https://api.within.co/api/v1/analyze/sentiment/batch"
    
    payload = {
        "items": [
            {
                "id": ad["id"],
                "text": ad["text"],
                "context": ad.get("context", {})
            }
            for ad in ads
        ],
        "settings": {
            "include_aspects": True,
            "include_sentence_level": True
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

## Limitations and Considerations

The sentiment analyzer has several known limitations:

### Content Limitations

- **Language Support**: Primary focus on English with limited support for other languages
- **Cultural Nuances**: May miss culturally-specific sentiment expressions
- **Emerging Terms**: May lag on very recent advertising terminology
- **Complex Irony**: Sometimes misses complex forms of irony or sarcasm

### Technical Limitations

- **Text Length**: Optimal performance for texts between 10-500 words
- **Processing Time**: Complex aspect analysis increases processing time
- **Confidence Reporting**: Lower confidence for unusual ad formats
- **Resource Usage**: Higher resource requirements than simpler sentiment analyzers

### Deployment Considerations

- **Latency Requirements**: ~150ms average for standard analysis
- **Batch Processing**: Recommended for analyzing large ad sets
- **Caching Strategy**: Consider caching for frequently analyzed content
- **Scale Requirements**: Plan for appropriate scaling during peak periods

## Future Improvements

The sentiment analyzer roadmap includes:

1. **Enhanced Multilingual Support**: Expanding robust support to more languages
2. **Multimodal Analysis**: Incorporating image sentiment alongside text
3. **More Granular Aspect Analysis**: Finer-grained aspect categorization
4. **Audience Segment Models**: Sentiment models for specific audience segments
5. **Emotion Intensity Calibration**: Improved calibration of emotion intensity
6. **Cultural Adaptation**: Better handling of cultural context
7. **Explanation Improvements**: More intuitive sentiment explanations
8. **Lighter Model Variants**: Smaller, faster models for edge deployment

For more information on related components, see:
- [NLP Pipeline Implementation](/docs/implementation/ml/nlp_pipeline.md)
- [Emotion Detection Implementation](/docs/implementation/ml/technical/emotion_detection.md)
- [Ad Sentiment Analyzer Model Card](/docs/implementation/ml/model_card_ad_sentiment_analyzer.md)
- [Sentiment Integration Guide](/docs/implementation/ml/integration/sentiment_integration.md) 