# Model Card: Ad Score Predictor

**IMPLEMENTATION STATUS: IMPLEMENTED**


## Model Details

- **Name**: Ad Score Predictor
- **Version**: 2.1.0
- **Type**: Hybrid (Gradient Boosting + Neural Network)
- **Purpose**: Predict ad effectiveness scores on a 0-100 scale
- **Creation Date**: 2023-01-15
- **Last Updated**: 2023-04-10

## Intended Use

### Primary Use Cases
- Predict effectiveness of ad content before campaign launch
- Compare potential ad variations to select the most effective
- Identify optimization opportunities for underperforming ads
- Benchmark ads against industry standards

### Out-of-Scope Uses
- Predicting exact CTR, conversion rates, or ROI (use specialized models)
- Content moderation or policy compliance checking
- Sentiment analysis of user comments or reviews
- Predicting performance for non-advertising content

### Target Users
- Advertisers and marketing teams
- Ad agencies and creative teams
- Marketing analysts and strategists
- Automated ad optimization systems

## Training Data

### Sources
- Historical ad performance data from Facebook, Google, Amazon, and TikTok
- Internal WITHIN customer ad campaigns (anonymized)
- Proprietary labeled dataset of ad effectiveness ratings
- Augmented with industry benchmark data

### Dataset Size
- 1.2 million ad examples
- Spanning January 2020 to December 2022
- Across 15 major industry verticals
- From 5 major advertising platforms

### Feature Distribution

| Feature Type | Count | Examples |
|--------------|-------|----------|
| Text features | 24 | Ad headline, description, CTA text |
| Numerical features | 18 | Historical CTR, word count, sentiment score |
| Categorical features | 12 | Industry, platform, audience type |
| Temporal features | 6 | Day of week, time of day, seasonality |

### Data Preparation
- Text cleaned and normalized using standard NLP pipeline
- Missing values imputed using appropriate strategies per feature
- Categorical features encoded using target encoding
- Numerical features normalized to zero mean, unit variance
- Stratified sampling to balance across industries and platforms

## Model Architecture

### Algorithm Type
Hybrid model combining:
- Gradient Boosted Decision Trees (XGBoost)
- Deep Neural Network with attention mechanisms

### Architecture Details
- **Tree Component**: XGBoost with 500 trees, max depth 8
- **Neural Component**:
  - Text embedding layer using domain-adapted BERT
  - 3 hidden layers (512, 256, 128 units)
  - Dropout (0.3) for regularization
  - BatchNorm after each hidden layer
  - Attention mechanism for feature importance

### Feature Inputs
- **Text Inputs**: 
  - Ad headline (tokenized, max 50 tokens)
  - Ad description (tokenized, max 200 tokens)
  - Call to action text (tokenized, max 10 tokens)
- **Numerical Inputs**:
  - Historical performance metrics (if available)
  - Text metadata (length, readability scores)
  - Audience match scores
- **Categorical Inputs**:
  - Ad platform (Facebook, Google, etc.)
  - Industry vertical
  - Ad format type
  - Target audience segments

### Output Format
- Primary output: Effectiveness score (0-100 scale)
- Secondary outputs:
  - Confidence score (0-1)
  - Component scores (content quality, audience match, etc.)
  - Feature importance values

## Performance Metrics

| Metric | Overall | Facebook | Google | Amazon | TikTok |
|--------|---------|----------|--------|--------|--------|
| RMSE | 8.6 | 7.9 | 8.2 | 9.3 | 10.1 |
| MAE | 6.4 | 5.8 | 6.1 | 7.2 | 7.8 |
| R² | 0.83 | 0.86 | 0.85 | 0.79 | 0.76 |
| Correlation with CTR | 0.78 | 0.82 | 0.79 | 0.75 | 0.71 |
| Processing Speed | 125ms | 110ms | 115ms | 130ms | 140ms |

## Limitations and Biases

### Known Limitations
- Performance decreases for highly innovative or unique ad formats
- Less accurate for industries with limited training data (e.g., niche B2B)
- Effectiveness predictions may not translate directly to performance metrics
- Requires minimum text content (very short ads may get lower confidence)
- Limited ability to assess visual elements beyond basic metadata

### Potential Biases
- Slight bias towards certain ad writing styles prevalent in training data
- May favor conventional ad structures over experimental approaches
- Performance varies by language, with highest accuracy for English
- May not fully account for cultural differences in advertising effectiveness

### Evaluation by Segment

| Segment | RMSE | R² | Notes |
|---------|------|-----|-------|
| E-commerce | 7.5 | 0.87 | Strong performance across platforms |
| B2B Services | 10.2 | 0.74 | Less training data, more variable |
| Entertainment | 8.1 | 0.83 | Good performance, especially social |
| Finance | 9.5 | 0.78 | Regulatory language impacts accuracy |
| Non-English | 11.8 | 0.70 | Lower performance, especially Asian languages |

## Ethical Considerations

### Data Privacy
- All training data is anonymized to remove advertiser identifiers
- No personally identifiable information (PII) used in training
- Performance metrics aggregated at industry level to protect confidentiality
- Model doesn't retain or expose individual ad data after training

### Fairness Assessment
- Regular bias audits conducted across industry, platform, and language
- Balanced training data across industry verticals to prevent favoritism
- Performance monitoring across demographics to identify potential bias
- Model optimized to reduce performance disparities across segments

### Potential Risks
- May inadvertently reinforce existing advertising conventions
- Could create feedback loops if widely adopted within an industry
- Potential for misuse if scores are interpreted as guaranteed outcomes
- Over-reliance might reduce creative experimentation

## Usage Instructions

### Required Environment
- Python 3.9+ with PyTorch 1.10+ and scikit-learn 1.0+
- 4GB RAM minimum, 8GB recommended
- GPU optional but recommended for batch processing
- Docker container available with all dependencies

### Setup
```bash
# Install from PyPI
pip install within-ad-score-predictor

# Or use Docker
docker pull within/ad-score-predictor:2.1.0
docker run -p 8000:8000 within/ad-score-predictor:2.1.0
```

### Inference Example
```python
from within.models import AdScorePredictor

# Initialize predictor
predictor = AdScorePredictor(version="2.1.0")

# Prepare ad data
ad_data = {
    "headline": "Limited Time Offer: 20% Off All Products",
    "description": "Shop our entire collection and save with this exclusive discount.",
    "cta": "Shop Now",
    "platform": "facebook",
    "industry": "retail",
    "format": "image_ad",
    "audience": ["shoppers", "deal_seekers"]
}

# Generate prediction
result = predictor.predict(ad_data)

# Access results
score = result["score"]  # Overall effectiveness score
confidence = result["confidence"]  # Prediction confidence
components = result["components"]  # Component scores
importance = result["importance"]  # Feature importance
```

### API Reference
Full API documentation is available at:
- [Python SDK Documentation](../../api/python_sdk.md)
- [REST API Documentation](../../api/endpoints.md#ad-score)

## Maintenance

### Owner
ML Engineering Team (ml-engineering@within.co)

### Update Frequency
- Major version updates: Quarterly
- Minor improvements: Monthly
- Monitoring: Continuous

### Monitoring Plan
- Daily drift detection for feature distributions
- Weekly performance evaluation on holdout sets
- Monthly comprehensive performance review
- Quarterly bias and fairness audit

### Retraining Triggers
- Performance degradation beyond 10% of baseline
- Feature drift beyond established thresholds
- Significant changes in ad platform capabilities
- Addition of substantial new training data

## Version History

| Version | Date | Changes | Performance Delta |
|---------|------|---------|-------------------|
| 1.0.0 | 2022-06-15 | Initial production model | Baseline |
| 1.1.0 | 2022-08-22 | Improved text preprocessing | +3.2% R² |
| 1.5.0 | 2022-11-10 | Added neural component to hybrid model | +7.5% R² |
| 2.0.0 | 2023-01-15 | Major architecture update with attention mechanism | +5.1% R² |
| 2.1.0 | 2023-04-10 | Performance optimizations and TikTok support | +1.2% R² |

## Supplementary Materials

- [Model Development Report](./ad_score_prediction.md)
- [Feature Engineering Documentation](./feature_engineering.md)
- [Model Evaluation Report](./model_evaluation.md)
- [Fairness Assessment](./fairness/ad_score_fairness.md)
- [Benchmarking Study](./benchmarks/ad_score_benchmarks.md) 