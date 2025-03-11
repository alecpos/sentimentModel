# Model Card: Ad Sentiment Analyzer

**IMPLEMENTATION STATUS: IMPLEMENTED**


## Model Details

- **Name**: Ad Sentiment Analyzer
- **Version**: 3.2.0
- **Type**: Fine-tuned Transformer (RoBERTa)
- **Purpose**: Analyze sentiment and emotional aspects of advertising text
- **Creation Date**: 2022-09-05
- **Last Updated**: 2023-07-10

## Intended Use

### Primary Use Cases
- Analyze sentiment of ad copy and headlines
- Detect emotional appeals in advertising content
- Measure audience reaction sentiment to ads
- Provide sentiment features for ad effectiveness scoring
- Compare sentiment across ad variations for optimization

### Out-of-Scope Uses
- General purpose sentiment analysis (not calibrated for non-advertising text)
- Content moderation or hate speech detection
- User comment classification (different domain)
- Detecting ad policy violations (different focus)
- Analysis of visual elements (image/video only)

### Target Users
- Advertising copywriters and creative teams
- Marketing analysts evaluating ad performance
- Ad optimization systems and tools
- WITHIN Ad Score Predictor (as a component)
- Campaign planning and testing tools

## Training Data

### Sources
- Advertising text from major platforms with human sentiment labels
- Customer ad copy with performance-based sentiment proxies
- Custom-labeled dataset of advertising headlines and descriptions
- Augmented with domain-adapted general sentiment datasets

### Dataset Size
- 500,000+ advertising text samples
- 50,000+ human-labeled sentiment annotations
- Spanning 20+ product/service categories
- Multiple ad formats (search, social, display, video)

### Feature Distribution

| Sentiment Category | Percentage | Examples |
|--------------------|------------|----------|
| Positive | 45% | Promotional, uplifting, solution-oriented |
| Neutral | 30% | Informational, descriptive, matter-of-fact |
| Negative | 15% | Problem-focused, pain-point highlighting |
| Mixed | 10% | Problem-solution format, contrasting emotions |

### Emotional Aspects

| Emotional Appeal | Examples |
|------------------|----------|
| Urgency | "Limited time offer", "Act now", "Don't miss out" |
| Trust | "Guaranteed results", "Trusted by millions", "Secure" |
| Joy | "Discover the joy", "Happiness awaits", "Enjoy more" |
| Fear | "Stop losing money", "Prevent disaster", "Avoid mistakes" |
| Curiosity | "The secret to", "Discover how", "You won't believe" |

### Data Preparation
- Text cleaning and normalization specific to ad content
- Handling of special characters and emojis common in ads
- Balanced sampling across sentiment categories
- Domain-specific augmentation techniques
- Cross-validation split by industry vertical to prevent leakage

## Model Architecture

### Algorithm Type
- Fine-tuned RoBERTa transformer model
- Advertising-specific sentiment classification head
- Multi-task learning for aspect-based sentiment analysis

### Architecture Details
- **Base Model**: RoBERTa-base (125M parameters)
- **Fine-tuning**: Full model fine-tuning with discriminative learning rates
- **Sequence Length**: 256 tokens maximum
- **Classification Heads**:
  - Main sentiment head (4 classes: positive, neutral, negative, mixed)
  - 5 aspect-specific sentiment heads (urgency, trust, joy, fear, curiosity)
  - Intensity prediction head (0-1 continuous scale)

### Feature Inputs
- **Raw Text**: Ad headlines, descriptions, and CTA text
- **Ad Context**: Ad type, platform, industry (optional)
- **Tokenization**: RoBERTa tokenizer with advertising-specific vocabulary extensions

### Output Format
- **Primary Output**: Sentiment classification with probabilities
  ```json
  {
    "sentiment": "positive",
    "confidence": 0.92,
    "intensity": 0.78
  }
  ```
- **Aspect-Based Output**: Emotional aspect scores
  ```json
  {
    "aspects": {
      "urgency": 0.85,
      "trust": 0.65,
      "joy": 0.72,
      "fear": 0.12,
      "curiosity": 0.45
    }
  }
  ```
- **Advanced Output**: Token-level sentiment contributions (for explainability)

## Performance Metrics

| Metric | Overall | Headlines | Descriptions | CTAs |
|--------|---------|-----------|--------------|------|
| Accuracy | 0.89 | 0.91 | 0.87 | 0.93 |
| Macro F1 | 0.87 | 0.88 | 0.84 | 0.90 |
| Precision | 0.88 | 0.89 | 0.86 | 0.92 |
| Recall | 0.87 | 0.87 | 0.83 | 0.89 |
| Processing Speed | 45ms | 30ms | 60ms | 25ms |

### Aspect-Based Performance

| Aspect | F1 Score | Precision | Recall |
|--------|----------|-----------|--------|
| Urgency | 0.92 | 0.93 | 0.91 |
| Trust | 0.86 | 0.87 | 0.85 |
| Joy | 0.89 | 0.90 | 0.88 |
| Fear | 0.91 | 0.92 | 0.90 |
| Curiosity | 0.87 | 0.88 | 0.86 |

## Limitations and Biases

### Known Limitations
- English language focus with limited multilingual capabilities
- Reduced performance on highly technical or specialized industry terminology
- May struggle with complex sarcasm or subtle messaging
- Limited context window (256 tokens) truncates very long ad copy
- Calibrated for digital advertising; less accurate for other domains

### Potential Biases
- Higher accuracy for e-commerce and retail than B2B or technical services
- Tendency to classify problem-solution formats as mixed rather than positive
- Slight bias toward detecting western cultural emotional expressions
- Some emoji combinations may be misinterpreted due to training data gaps
- May underdetect sentiment in highly abbreviated or jargon-heavy text

### Evaluation by Segment

| Segment | Accuracy | F1 Score | Notes |
|---------|----------|----------|-------|
| E-commerce | 0.92 | 0.90 | Strong performance on promotional language |
| B2B Tech | 0.83 | 0.80 | Technical jargon sometimes misclassified |
| Financial | 0.86 | 0.84 | Regulatory language impacts detection |
| Health/Wellness | 0.88 | 0.86 | Strong on emotional appeals |
| Travel/Leisure | 0.91 | 0.89 | Good at aspiration/excitement detection |
| Non-English | 0.77 | 0.73 | Limited multilingual capabilities |

## Ethical Considerations

### Data Privacy
- No personally identifiable information used in training
- Ad text is anonymized and advertisers are de-identified
- Model does not memorize specific ad content
- Output data is handled according to data governance policies

### Fairness Assessment
- Evaluated for consistent performance across industries
- Tested for cultural and regional expression bias
- Regular audits for gender and demographic term sensitivity
- Continual monitoring for emergent biases

### Potential Risks
- May reinforce certain advertising sentiment norms
- Could influence standardization of ad emotional appeals
- Potential misinterpretation of culturally-specific expressions
- Risk of over-optimizing toward specific sentiment patterns

## Usage Instructions

### Required Environment
- Python 3.9+ with PyTorch 1.10+ and transformers 4.15+
- 4GB RAM minimum, 2GB GPU VRAM recommended for batch processing
- CPU-only mode available with performance penalty
- Docker container available with all dependencies

### Setup
```bash
# Install from PyPI
pip install within-ad-sentiment-analyzer

# Or use Docker
docker pull within/ad-sentiment-analyzer:3.2.0
docker run -p 8002:8002 within/ad-sentiment-analyzer:3.2.0
```

### Inference Example
```python
from within.nlp import AdSentimentAnalyzer

# Initialize analyzer
analyzer = AdSentimentAnalyzer(version="3.2.0")

# Single text analysis
sentiment = analyzer.analyze(
    text="Limited time offer! Get 50% off our premium package and transform your results today.",
    ad_type="promotional",
    include_aspects=True
)

# Batch analysis
texts = [
    "Upgrade your experience with our premium service.",
    "Stop wasting money on ineffective solutions. Try our proven system.",
    "The industry's most reliable platform for professionals."
]
batch_results = analyzer.analyze_batch(texts, include_aspects=True)

# Access results
print(f"Sentiment: {sentiment['sentiment']}")
print(f"Confidence: {sentiment['confidence']}")
print(f"Urgency aspect: {sentiment['aspects']['urgency']}")
```

### API Reference
Full API documentation is available at:
- [Python SDK Documentation](../../api/python_sdk.md)
- [REST API Documentation](../../api/endpoints.md#sentiment-analysis)

## Maintenance

### Owner
NLP Engineering Team (nlp-engineering@within.co)

### Update Frequency
- Major version updates: Bi-annually
- Minor improvements: Monthly
- Monitoring: Continuous

### Monitoring Plan
- Daily performance metrics monitoring
- Weekly prediction distribution analysis
- Monthly manual evaluation on sample data
- Quarterly comprehensive review and bias testing

### Retraining Triggers
- Performance degradation beyond established thresholds
- Significant changes in digital advertising language patterns
- Addition of new industry domains or ad formats
- Expansion to new languages or markets
- Substantial new labeled data availability (>10% of original dataset)

## Version History

| Version | Date | Changes | Performance Delta |
|---------|------|---------|-------------------|
| 1.0.0 | 2021-06-20 | Initial BERT-based model | Baseline |
| 2.0.0 | 2022-02-15 | Migrated to RoBERTa architecture | +4.8% Accuracy |
| 3.0.0 | 2022-09-05 | Added aspect-based sentiment analysis | +Multiple metrics |
| 3.1.0 | 2023-01-12 | Improved token attribution for explainability | No accuracy change |
| 3.2.0 | 2023-07-10 | Enhanced emotional aspect detection, added intensity scoring | +3.2% Aspect F1 |

## Supplementary Materials

- [NLP Pipeline Documentation](./nlp_pipeline.md)
- [Sentiment Analysis Methodology](./technical/sentiment_analysis.md)
- [Emotion Detection in Advertising Text](./technical/emotion_detection.md)
- [Model Evaluation Report](./evaluation/sentiment_analyzer_evaluation.md)
- [Integration Guide](./integration/sentiment_integration.md) 