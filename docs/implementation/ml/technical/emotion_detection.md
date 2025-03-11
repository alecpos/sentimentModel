# Emotion Detection in Ad Text

**IMPLEMENTATION STATUS: IMPLEMENTED**


This document describes the emotion detection methodology implemented in the WITHIN platform for analyzing emotional content in advertising text.

## Table of Contents

- [Overview](#overview)
- [Emotion Framework](#emotion-framework)
- [Technical Implementation](#technical-implementation)
- [Model Architecture](#model-architecture)
- [Training Data](#training-data)
- [Feature Engineering](#feature-engineering)
- [Emotion Recognition Process](#emotion-recognition-process)
- [Contextual Adaptation](#contextual-adaptation)
- [Performance Metrics](#performance-metrics)
- [Integration Guidelines](#integration-guidelines)
- [Limitations and Considerations](#limitations-and-considerations)
- [Future Enhancements](#future-enhancements)

## Overview

The WITHIN Emotion Detection system identifies and quantifies emotional content in advertising text across multiple dimensions. This capability enables advertisers to understand the emotional impact of their messaging and optimize ad content for specific emotional responses.

### Core Capabilities

- Detection of 8 primary emotional dimensions in ad text
- Measurement of emotion intensity (strength)
- Identification of mixed emotions and emotional transitions
- Industry and audience-specific emotional calibration
- Integration with ad effectiveness prediction
- Explanations for emotion detection results

## Emotion Framework

The emotion detection system is based on a dimensional model of emotions calibrated specifically for advertising content.

### Primary Emotion Dimensions

The system analyzes 8 primary emotional dimensions:

1. **Joy/Happiness**: Feelings of pleasure, happiness, and satisfaction
2. **Trust/Reliability**: Feelings of confidence, security, and dependability
3. **Fear/Anxiety**: Feelings of apprehension, worry, or concern
4. **Surprise/Amazement**: Feelings of astonishment or unexpectedness
5. **Sadness/Disappointment**: Feelings of loss, disappointment, or lack
6. **Anger/Frustration**: Feelings of irritation, annoyance, or hostility
7. **Anticipation/Excitement**: Feelings of looking forward to something
8. **Disgust/Displeasure**: Feelings of revulsion or strong dislike

### Emotion Intensity

Each emotion is measured on a continuous scale from 0.0 (absent) to 1.0 (strongly present), rather than as binary categories, allowing for nuanced emotional profiles.

### Emotional Combinations

The system recognizes that ad text often contains multiple emotions simultaneously, enabling the detection of complex emotional patterns like:

- **Anticipatory Joy**: Excitement combined with happiness
- **Anxious Anticipation**: Anticipation mixed with mild fear
- **Bittersweet**: Combination of sadness and happiness
- **Intrigued Surprise**: Surprise combined with anticipation

### Emotion Wheels

The emotion detection system is based on a modified version of Plutchik's emotion wheel, adapted specifically for advertising context:

```
                Joy
                 │
    Anticipation ┌┴┐ Trust
               ╱   ╲
              ╱     ╲
             ╱       ╲
            ╱         ╲
   Surprise │         │ Fear
            ╲         ╱
             ╲       ╱
              ╲     ╱
               ╲   ╱
     Disgust   └┬┘ Anger
                 │
               Sadness
```

## Technical Implementation

The emotion detection system uses a multi-faceted technical approach.

### System Architecture

```
┌─────────────────────────┐
│      Ad Text Input      │
└───────────┬─────────────┘
            │
┌───────────▼─────────────┐
│  Text Preprocessing     │
└───────────┬─────────────┘
            │
┌───────────▼─────────────┐
│  Language Representation│
└───────────┬─────────────┘
            │
         ┌──┴──┐
         │     │
┌────────▼───┐ │ ┌─────────▼────────┐
│ Deep       │ │ │ Lexical Emotion  │
│ Learning   │ │ │ Detection        │
│ Pipeline   │ │ │                  │
└────────┬───┘ │ └─────────┬────────┘
         │     │           │
         │  ┌──▼───────────▼──┐
         │  │ Ensemble        │
         └──►  Integration    │
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │ Contextual      │
            │ Calibration     │
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │ Multi-emotion   │
            │ Analysis        │
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │ Output          │
            │ Formatting      │
            └─────────────────┘
```

### Implementation Approach

The emotion detection system uses a hybrid approach:

1. **Deep Learning Pipeline**:
   - Transformer-based emotion classification
   - Sequence labeling for emotion spans
   - Attention visualization for explanation

2. **Lexical Emotion Detection**:
   - Advertising-specific emotion lexicons
   - Emotion intensity modifiers
   - Negation handling

3. **Ensemble Integration**:
   - Model weighting based on confidence
   - Complementary strength utilization
   - Confidence estimation

4. **Contextual Calibration**:
   - Industry-specific emotion calibration
   - Platform-specific emotion normalization
   - Audience-specific emotion adjustment

5. **Multi-emotion Analysis**:
   - Emotion co-occurrence patterns
   - Emotional progression analysis
   - Dominant emotion identification

## Model Architecture

The deep learning component of the emotion detection system uses a specialized neural architecture.

### Base Architecture

The system is built on a transformer architecture with emotion-specific modifications:

- **Base Model**: RoBERTa with advertising domain adaptation
- **Modification**: Multi-head classification architecture
- **Specialization**: Emotion-specific attention mechanisms

### Emotion Classification Heads

```
                   Input Text
                       │
                       ▼
                   Tokenization
                       │
                       ▼
                 RoBERTa Encoder
                       │
                       ▼
             Contextual Representations
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
  ┌─────────┐     ┌─────────┐     ┌─────────┐
  │ Emotion │     │ Emotion │     │ Emotion │
  │ Head 1  │     │ Head 2  │ ... │ Head 8  │
  │(Joy)    │     │(Trust)  │     │(Disgust)│
  └─────────┘     └─────────┘     └─────────┘
       │               │               │
       └───────────────┼───────────────┘
                       │
                       ▼
              Emotion Profile Vector
                       │
                       ▼
              Confidence Estimation
```

Each emotion head is trained to detect a specific emotional dimension, enabling the system to recognize multiple emotions simultaneously.

### Technical Specifications

- **Input Sequence Length**: Up to 256 tokens
- **Hidden Layer Size**: 768
- **Attention Heads**: 12
- **Classification Heads**: 8 (one per primary emotion)
- **Parameters**: 135M total parameters
- **Output Format**: Probability scores for each emotion (0.0-1.0)

### Training Methodology

The model was trained using a multi-task learning approach:

1. **Domain Adaptation**: Further pre-training on advertising corpus
2. **Emotion Classification**: Multi-label classification for 8 emotions
3. **Intensity Regression**: Regression for emotion intensity
4. **Span Detection**: Token-level tagging for emotion spans

## Training Data

The emotion detection model was trained on a specialized dataset of advertising content.

### Dataset Composition

- **Total Examples**: 325,000 annotated ad texts
- **Platforms**: Facebook, Google, Instagram, LinkedIn, TikTok, Twitter
- **Industries**: 35+ distinct industry categories
- **Time Range**: 2018-2023
- **Languages**: Primary focus on English

### Annotation Process

The training data was annotated through a multi-stage process:

1. **Expert Annotation**: Initial annotation by advertising psychology experts
2. **Multi-annotator Agreement**: Each example annotated by 3+ annotators
3. **Reconciliation**: Disagreements resolved through expert review
4. **Calibration**: Periodic calibration sessions for annotators
5. **Audience Validation**: Sample validation with target demographic panels

### Distribution of Emotions

| Emotion | Frequency | Avg. Intensity |
|---------|-----------|---------------|
| Joy/Happiness | 48.5% | 0.67 |
| Trust/Reliability | 52.3% | 0.71 |
| Fear/Anxiety | 18.7% | 0.58 |
| Surprise/Amazement | 27.9% | 0.63 |
| Sadness/Disappointment | 12.4% | 0.51 |
| Anger/Frustration | 9.8% | 0.55 |
| Anticipation/Excitement | 43.2% | 0.69 |
| Disgust/Displeasure | 7.3% | 0.48 |

*Note: Percentages sum to >100% because ads often contain multiple emotions*

### Annotation Quality

- **Inter-annotator Agreement**: Cohen's Kappa of 0.79
- **Emotion Intensity Correlation**: 0.85 Pearson correlation between annotators
- **Quality Monitoring**: Regular quality checks throughout annotation process

## Feature Engineering

The emotion detection system leverages multiple feature types beyond raw text.

### Linguistic Features

- Lexical emotion markers
- Part-of-speech patterns
- Syntactic structures
- Rhetorical devices
- Punctuation patterns
- Capitalization usage

### Semantic Features

- Semantic role labeling
- Entity sentiment relationships
- Temporal expressions
- Causal relationships
- Comparative structures
- Hypothetical constructions

### Advertising-Specific Features

- Call-to-action emotional content
- Benefit statement emotional framing
- Promotional language emotional markers
- Brand-related emotional associations
- Product feature emotional connections
- Social proof emotional elements

### Feature Importance

Analysis of feature importance shows varying contribution by feature type:

```
                     Feature Importance by Emotion
100% ┼───────────────────────────────────────────────────────────┐
     │                                                           │
     │    ■ Lexical   □ Syntactic   ▨ Semantic   ▤ Ad-specific   │
 80% ┤                                                           │
     │                                                           │
     │                                                           │
 60% ┤                                                           │
     │                                                           │
     │                                                           │
 40% ┤                                                           │
     │                                                           │
     │                                                           │
 20% ┤                                                           │
     │                                                           │
     │                                                           │
  0% ┼───┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┤
      Joy Trust Fear Surp Sad  Ang  Ant  Dis  Mixed
```

## Emotion Recognition Process

The emotion recognition process involves multiple stages of analysis.

### Processing Flow

1. **Text Preprocessing**:
   - Normalization and tokenization
   - Ad-specific preprocessing
   - Sentence and phrase segmentation

2. **Emotion Detection**:
   - Independent scoring for each emotion dimension
   - Identification of emotion-bearing phrases
   - Confidence scoring for each detection

3. **Intensity Determination**:
   - Analysis of intensifiers and modifiers
   - Context-based intensity adjustment
   - Relative intensity calculation

4. **Multi-emotion Analysis**:
   - Emotion co-occurrence detection
   - Dominant emotion identification
   - Emotional progression through text

5. **Contextual Calibration**:
   - Industry-specific normalization
   - Audience targeting adjustments
   - Platform-specific calibration

### Emotion Span Detection

The system identifies specific spans of text that convey each emotion:

```python
def detect_emotion_spans(text, emotions=None):
    """Detect text spans conveying specific emotions.
    
    Args:
        text: The ad text to analyze
        emotions: List of emotions to detect, or None for all
        
    Returns:
        Dictionary mapping emotions to lists of text spans
    """
    # Preprocess text
    tokens, features = preprocess_for_span_detection(text)
    
    # Get emotions to detect
    emotions = emotions or ALL_EMOTIONS
    
    # Run span detection model
    spans = {}
    model_outputs = emotion_span_model(tokens, features)
    
    # Process model outputs for each emotion
    for emotion in emotions:
        emotion_spans = extract_spans_for_emotion(
            tokens, model_outputs, emotion
        )
        
        if emotion_spans:
            spans[emotion] = emotion_spans
    
    return spans
```

### Emotion Intensity Analysis

The system analyzes modifiers that affect emotion intensity:

```python
def analyze_intensity_modifiers(text, emotion_spans):
    """Analyze modifiers affecting emotion intensity.
    
    Args:
        text: The ad text
        emotion_spans: Detected emotion spans
        
    Returns:
        Updated emotion spans with intensity information
    """
    # Parse text for modifiers
    parsed = dependency_parser(text)
    
    # Extract intensifiers and diminishers
    modifiers = extract_intensity_modifiers(parsed)
    
    # Apply modifiers to emotion spans
    for emotion, spans in emotion_spans.items():
        for i, span in enumerate(spans):
            # Find modifiers that apply to this span
            span_modifiers = find_applicable_modifiers(
                span, modifiers, parsed
            )
            
            # Calculate intensity adjustment
            intensity_multiplier = calculate_intensity_adjustment(
                span_modifiers
            )
            
            # Apply adjustment to base intensity
            spans[i]["intensity"] *= intensity_multiplier
            
            # Record modifiers for explanation
            spans[i]["modifiers"] = span_modifiers
    
    return emotion_spans
```

## Contextual Adaptation

The emotion detection system adapts to different contexts for improved accuracy.

### Industry Context

Different industries have distinct emotional profiles and expectations:

- **E-commerce**: Higher baseline for anticipation/excitement
- **Financial Services**: Higher sensitivity to trust/security emotions
- **Healthcare**: Different calibration for fear/concern expressions
- **Entertainment**: Different thresholds for joy/excitement

### Platform Context

Emotional expression varies by advertising platform:

- **LinkedIn**: More formal emotional expression
- **Instagram**: More visually-oriented emotional cues
- **TikTok**: Higher intensity emotional expressions
- **Facebook**: Broader range of emotional expressions

### Audience Context

Emotional perception varies by target audience:

- **Demographic Factors**: Age, gender, culture
- **Psychographic Factors**: Values, interests, lifestyle
- **Purchase Intent Stage**: Awareness vs. consideration vs. conversion

### Contextual Adaptation Implementation

```python
def apply_contextual_calibration(emotion_scores, context):
    """Apply contextual calibration to emotion scores.
    
    Args:
        emotion_scores: Raw emotion scores
        context: Dictionary with context information
        
    Returns:
        Calibrated emotion scores
    """
    calibrated = emotion_scores.copy()
    
    # Apply industry calibration
    if "industry" in context:
        industry_calibration = get_industry_calibration(
            context["industry"]
        )
        calibrated = apply_calibration(
            calibrated, industry_calibration
        )
    
    # Apply platform calibration
    if "platform" in context:
        platform_calibration = get_platform_calibration(
            context["platform"]
        )
        calibrated = apply_calibration(
            calibrated, platform_calibration
        )
    
    # Apply audience calibration
    if "audience" in context:
        audience_calibration = get_audience_calibration(
            context["audience"]
        )
        calibrated = apply_calibration(
            calibrated, audience_calibration
        )
    
    return calibrated
```

## Performance Metrics

The emotion detection system has been extensively evaluated on multiple datasets.

### Overall Performance

| Metric | Score |
|--------|-------|
| Macro F1 | 0.823 |
| Micro F1 | 0.862 |
| Weighted F1 | 0.835 |
| Hamming Loss | 0.143 |
| Exact Match Ratio | 0.681 |

### Per-Emotion Performance

| Emotion | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
| Joy/Happiness | 0.875 | 0.842 | 0.858 |
| Trust/Reliability | 0.863 | 0.824 | 0.843 |
| Fear/Anxiety | 0.815 | 0.762 | 0.788 |
| Surprise/Amazement | 0.829 | 0.791 | 0.810 |
| Sadness/Disappointment | 0.782 | 0.733 | 0.757 |
| Anger/Frustration | 0.768 | 0.721 | 0.744 |
| Anticipation/Excitement | 0.858 | 0.837 | 0.847 |
| Disgust/Displeasure | 0.755 | 0.704 | 0.729 |

### Intensity Accuracy

| Emotion | MAE | RMSE | Corr. |
|---------|-----|------|-------|
| Joy/Happiness | 0.112 | 0.156 | 0.831 |
| Trust/Reliability | 0.124 | 0.167 | 0.815 |
| Fear/Anxiety | 0.138 | 0.183 | 0.782 |
| Surprise/Amazement | 0.129 | 0.172 | 0.794 |
| Sadness/Disappointment | 0.142 | 0.189 | 0.768 |
| Anger/Frustration | 0.147 | 0.195 | 0.754 |
| Anticipation/Excitement | 0.118 | 0.159 | 0.825 |
| Disgust/Displeasure | 0.153 | 0.201 | 0.741 |

*MAE: Mean Absolute Error, RMSE: Root Mean Square Error, Corr: Correlation with human annotations*

### Performance by Industry

| Industry | Macro F1 | Micro F1 |
|----------|----------|----------|
| E-commerce | 0.847 | 0.881 |
| Financial Services | 0.831 | 0.865 |
| Technology | 0.842 | 0.874 |
| Healthcare | 0.812 | 0.849 |
| Travel | 0.834 | 0.868 |
| Entertainment | 0.839 | 0.872 |

### Comparative Evaluation

The system was compared to general-purpose emotion detection systems:

| System | Macro F1 | Micro F1 |
|--------|----------|----------|
| WITHIN Emotion Detection | 0.823 | 0.862 |
| General Emotion Detector A | 0.745 | 0.782 |
| General Emotion Detector B | 0.762 | 0.796 |
| Previous WITHIN Version | 0.789 | 0.834 |

## Integration Guidelines

The emotion detection system can be integrated with other systems through multiple interfaces.

### API Integration

```python
import requests
import json

def detect_emotions(ad_text, api_key, context=None):
    """Detect emotions in ad text using the WITHIN API."""
    url = "https://api.within.co/api/v1/analyze/emotions"
    
    payload = {
        "text": ad_text,
        "context": context or {
            "industry": "general",
            "platform": "facebook",
            "audience": {"demographic": "general"}
        },
        "settings": {
            "include_spans": True,
            "threshold": 0.15  # Min score to include emotion
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

# Analyze emotions
result = client.analyze_emotions(
    text="Experience the joy of unlimited streaming with our premium plan!",
    context={
        "industry": "entertainment",
        "platform": "facebook",
        "audience": {"demographic": "general"}
    }
)

# Process results
print("Detected emotions:")
for emotion, data in result.emotions.items():
    if data.score > 0.2:  # Threshold for significant emotions
        print(f"- {emotion}: {data.score:.2f}")
        for span in data.spans:
            print(f"  '{span.text}' ({span.intensity:.2f})")
```

### Batch Processing

For analyzing multiple ad texts:

```python
def batch_emotion_analysis(ad_texts, api_key, context=None):
    """Analyze emotions in multiple ad texts."""
    url = "https://api.within.co/api/v1/analyze/emotions/batch"
    
    payload = {
        "items": [
            {
                "id": str(i),
                "text": text,
                "context": context or {
                    "industry": "general", 
                    "platform": "facebook"
                }
            }
            for i, text in enumerate(ad_texts)
        ],
        "settings": {
            "include_spans": True,
            "threshold": 0.15  # Min score to include emotion
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

The emotion detection system has several known limitations:

### Content Limitations

- **Language Support**: Primary focus on English with limited support for other languages
- **Cultural Context**: May miss culturally-specific emotional expressions
- **Figurative Language**: May misinterpret complex metaphors or idiomatic expressions
- **Subtle Emotions**: Less accurate with subtle or mixed emotional expressions

### Technical Limitations

- **Text Length**: Optimal performance for texts between 10-500 words
- **Novel Expressions**: May miss very recent emotional expressions or slang
- **Ambiguous Content**: Lower confidence scores for ambiguous emotional content
- **Platform-Specific Features**: Different accuracy levels across advertising platforms

### Deployment Considerations

- **Processing Time**: ~75-150ms per analysis depending on text length
- **Batch Processing**: More efficient for analyzing multiple texts
- **Threshold Selection**: Consider adjusting confidence thresholds based on use case
- **Contextual Information**: Providing accurate context improves performance

## Future Enhancements

Planned enhancements to the emotion detection system include:

1. **Multimodal Emotion Analysis**: Integrating text and image emotion detection
2. **Enhanced Multilingual Support**: Expanding robust detection to more languages
3. **Cultural Adaptation**: Better handling of cultural variations in emotional expression
4. **Micro-Emotion Detection**: Identification of more nuanced emotional states
5. **Emotional Journey Mapping**: Analyzing emotional progression through longer content
6. **Personalized Emotion Models**: Customized for specific audience segments
7. **Real-time Drift Detection**: Adapting to evolving emotional expression patterns
8. **Lightweight Model Variants**: Optimized models for edge deployment

For more information about related components, see:
- [NLP Pipeline Implementation](/docs/implementation/ml/nlp_pipeline.md)
- [Sentiment Analysis Methodology](/docs/implementation/ml/technical/sentiment_analysis.md)
- [Ad Sentiment Analyzer Model Card](/docs/implementation/ml/model_card_ad
_sentiment_analyzer.md)
- [Sentiment Integration Guide](/docs/implementation/ml/integration/sentiment_integration.md) 