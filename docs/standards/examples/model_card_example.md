# ML Model Card: Ad Click-Through Rate Predictor

**IMPLEMENTATION STATUS: IMPLEMENTED**


## Model Overview
- **Model Name:** Ad Click-Through Rate Predictor
- **Version:** 2.1.0
- **Type:** Gradient Boosted Decision Tree Classification
- **Purpose:** Predict the likelihood of users clicking on specific digital advertisements
- **Created Date:** 2023-01-15
- **Last Updated:** 2023-09-05

## Intended Use
- **Primary Use Cases:** 
  - Optimize ad placement and targeting
  - Forecast campaign performance
  - Automate bid adjustments
  - Identify high-potential ad creative elements

- **Out-of-Scope Uses:**
  - User behavioral profiling beyond advertising context
  - Credit scoring or financial decision-making
  - Demographic classification of individuals
  - Medical or health-related predictions

- **Target Users:**
  - Digital marketing specialists
  - Campaign managers
  - Media buyers
  - Marketing analytics teams

## Training Data
- **Dataset Sources:**
  - WITHIN Ad Platform click logs (Jan 2022 - Jun 2023)
  - Anonymized campaign performance data (5,000+ campaigns)
  - Synthetic minority samples for rare categories (10% of dataset)

- **Dataset Size:** 150 million click events

- **Feature Distribution:**
  - Click outcome: 2.7% positive (clicked), 97.3% negative (not clicked)
  - Ad categories: Retail (45%), Technology (25%), Financial (15%), Other (15%)
  - Device types: Mobile (65%), Desktop (30%), Tablet (5%)
  - Time distribution: Balanced across hours and days of week

- **Data Preparation:**
  - Feature encoding with target encoding for categorical variables
  - Time features extracted (hour, day, month, holiday flags)
  - Class imbalance addressed with SMOTE and weighted sampling
  - Feature scaling with robust scaler to handle outliers

## Model Architecture
- **Algorithm Type:** LightGBM Gradient Boosted Decision Trees
- **Architecture Details:**
  - 500 trees with max depth of 8
  - Learning rate: 0.05
  - L1 regularization: 0.2
  - L2 regularization: 0.5
  - Early stopping after 50 rounds without improvement
  - Feature interaction constraints based on domain knowledge

- **Feature Inputs:**
  - Ad creative features (50 features including text embeddings)
  - User context features (15 features)
  - Campaign configuration features (25 features)
  - Temporal features (10 features)

- **Output Format:**
  - Probability score (0.0-1.0) representing click likelihood
  - Binary classification (clicked/not clicked) with 0.5 threshold
  - Prediction confidence score
  - Top contributing features

## Performance Metrics
| Metric | Value | Notes |
|--------|-------|-------|
| AUC-ROC | 0.86 | Stable across product categories |
| Precision | 0.12 | At default threshold of 0.5 |
| Recall | 0.73 | At default threshold of 0.5 |
| F1 Score | 0.21 | At default threshold of 0.5 |
| Log Loss | 0.17 | Indicates good probability calibration |
| Inference Time | 12ms | Average per prediction on production hardware |

## Limitations and Biases
- **Known Limitations:**
  - Performance degrades for very new ad categories with limited data
  - Less accurate during major shopping events (Black Friday, Cyber Monday)
  - Limited effectiveness for video ad formats
  - Requires retraining at least quarterly to maintain performance

- **Potential Biases:**
  - Slightly higher false positive rates for mobile users
  - Performance varies by geography (stronger in US/UK markets)
  - May amplify existing engagement patterns in historical data
  - Better performance for text-heavy ads than image-focused ads

- **Evaluation Results by Segment:**
  | Segment | AUC-ROC | Precision | Notes |
  |---------|-----------|-----------|-------|
  | Retail | 0.88 | 0.14 | Best performance across categories |
  | Financial | 0.84 | 0.09 | Lower precision due to category specifics |
  | Mobile | 0.85 | 0.11 | Slightly lower than average |
  | Desktop | 0.87 | 0.16 | Best performing device type |
  | New Users | 0.81 | 0.08 | Limited historical context |

## Ethical Considerations
- **Data Privacy:**
  - No personally identifiable information used in training
  - All user data anonymized and aggregated
  - Compliance with GDPR, CCPA, and other privacy regulations
  - Minimal data retention policy enforced

- **Fairness Assessment:**
  - Performance audited across demographic groups when available
  - No significant disparities found across age groups
  - Regular bias monitoring implemented in production
  - Fairness constraints incorporated into model objectives

- **Potential Risks:**
  - May reinforce existing content consumption patterns
  - Could optimize toward clickbait-style content if unconstrained
  - Requires monitoring to prevent inadvertent category bias
  - Potential system gaming by sophisticated adversaries

## Usage Instructions
- **Required Environment:**
  - Python 3.9+
  - LightGBM 3.3.0+
  - NumPy 1.20.0+
  - Pandas 1.3.0+
  - 4GB RAM minimum

- **Setup Steps:**
  ```python
  from within.models.ad_prediction import CTRPredictor
  
  # Initialize with default model
  predictor = CTRPredictor()
  
  # Or specify model version
  predictor = CTRPredictor(version="2.1.0")
  ```

- **Inference Examples:**
  ```python
  # Single prediction
  ad_features = {
      "ad_id": "ad_123456",
      "headline": "Summer Sale - 50% Off Everything",
      "description": "Limited time offer on all products. Shop now!",
      "campaign_id": "camp_789012",
      "product_category": "apparel",
      "device_type": "mobile",
      "hour_of_day": 14,
      "day_of_week": 2,
      # ... additional features
  }
  
  prediction = predictor.predict(ad_features)
  print(f"Click probability: {prediction['probability']:.4f}")
  print(f"Prediction: {'Click' if prediction['prediction'] == 1 else 'No click'}")
  print(f"Top features: {prediction['top_features']}")
  
  # Batch prediction
  ad_features_batch = [ad_features_1, ad_features_2, ad_features_3]
  predictions = predictor.predict_batch(ad_features_batch)
  ```

- **API Reference:** See `/docs/implementation/api/endpoints/ad_prediction.md` for detailed API documentation

## Maintenance
- **Owner:** WITHIN ML Team (ml-team@within.example.com)
- **Update Frequency:** Quarterly retraining with monthly monitoring
- **Monitoring Plan:**
  - Daily performance metrics tracking
  - Weekly drift detection
  - Monthly fairness audit
  - Quarterly deep-dive analysis

- **Retraining Triggers:**
  - AUC-ROC drops below 0.82 on monitoring dataset
  - Feature distribution drift exceeds 15%
  - Quarterly scheduled retraining regardless of performance
  - New major feature introduction

## Version History
| Version | Date | Changes | Performance Delta |
|---------|------|---------|-------------------|
| 1.0.0 | 2022-03-10 | Initial release | Baseline |
| 1.5.0 | 2022-06-22 | Added text embeddings, increased tree depth | +4.7% AUC-ROC |
| 2.0.0 | 2022-11-15 | Migrated to LightGBM, added temporal features | +3.2% AUC-ROC |
| 2.1.0 | 2023-09-05 | Retraining with new data, improved feature engineering | +1.1% AUC-ROC |

---

*This model card follows the WITHIN ML documentation standards and is based on the template available at `/docs/standards/templates/model_card_template.md`.* 