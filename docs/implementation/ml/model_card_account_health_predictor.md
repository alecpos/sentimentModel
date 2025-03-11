# Model Card: Account Health Predictor

**IMPLEMENTATION STATUS: IMPLEMENTED**


## Model Details

- **Name**: Account Health Predictor
- **Version**: 1.5.0
- **Type**: Ensemble (Time Series + Anomaly Detection)
- **Purpose**: Assess advertising account health and predict future issues
- **Creation Date**: 2022-12-10
- **Last Updated**: 2023-05-22

## Intended Use

### Primary Use Cases
- Monitor overall advertising account health
- Detect anomalies in account performance
- Identify risk factors that may impact performance
- Generate recommendations for account optimization
- Predict future performance trends

### Out-of-Scope Uses
- Individual ad performance prediction (use Ad Score Predictor)
- User behavior prediction outside advertising platforms
- Fraud detection (use specialized fraud models)
- Content moderation or policy compliance

### Target Users
- Account managers and strategists
- Marketing operations teams
- Performance marketing analysts
- Automated monitoring systems
- Agency-client reporting systems

## Training Data

### Sources
- Historical account performance data from major ad platforms
- Anonymized WITHIN customer account data
- Platform-specific API metrics and benchmarks
- Labeled dataset of account health states and interventions

### Dataset Size
- 50,000+ account snapshots
- 5,000+ unique advertising accounts
- 3+ years of historical data (2020-2023)
- Across 12 industry verticals

### Feature Distribution

| Feature Type | Count | Examples |
|--------------|-------|----------|
| Performance metrics | 35 | CTR, conversion rate, ROAS, CPA |
| Temporal patterns | 22 | Daily/weekly trends, volatility measures |
| Account settings | 18 | Budget utilization, campaign diversity |
| Structural features | 15 | Account organization, targeting breadth |
| Platform specifics | 12 | Quality scores, relevance metrics per platform |

### Data Preparation
- Time series normalization and detrending
- Outlier detection and treatment
- Feature scaling using robust scaler
- Missing data imputation via forward fill and model-based methods
- Temporal feature extraction (trends, seasonality, volatility)

## Model Architecture

### Algorithm Type
Multi-component ensemble consisting of:
- Time Series Models (Prophet, LSTM)
- Anomaly Detection (Isolation Forest, DBSCAN)
- Classification Models (Gradient Boosted Trees)
- Recommendation Engine (Association Rules + Ranking)

### Architecture Details
- **Time Series Component**: 
  - Facebook Prophet for trend/seasonality decomposition
  - LSTM networks for sequence modeling (3 layers, 128 units each)
  - Horizon: 30 days forward prediction
  
- **Anomaly Component**:
  - Isolation Forest for unsupervised anomaly detection
  - DBSCAN for density-based anomaly clustering
  - Multivariate anomaly scoring system
  
- **Health Classification**:
  - XGBoost classifier (500 trees, max depth 7)
  - 4-class health categorization (Excellent, Good, At Risk, Critical)
  - Calibrated probability outputs
  
- **Recommendation Engine**:
  - Association rule mining for intervention discovery
  - XGBoost ranker for recommendation prioritization
  - Impact estimation model for ROI calculation

### Feature Inputs
- **Time Series Data**: 
  - Daily performance metrics (last 90 days)
  - Weekly aggregated metrics (last 52 weeks)
  - Monthly trends (last 24 months)
  
- **Account Structure Data**:
  - Campaign count and diversity
  - Ad group distribution
  - Targeting settings
  - Budget allocation
  
- **Performance Metrics**:
  - Efficiency metrics (CPC, CPM, CTR)
  - Conversion metrics (CR, CPA, ROAS)
  - Engagement metrics (view rate, interaction rate)
  - Quality metrics (relevance scores, quality scores)
  
- **Platform-Specific Metrics**:
  - Platform quality signals
  - Audience overlap metrics
  - Competitive density
  - Platform-specific limitations

### Output Format
- **Primary Outputs**:
  - Health Score (0-100)
  - Health Classification (Excellent, Good, At Risk, Critical)
  - Confidence Score (0-1)
  
- **Secondary Outputs**:
  - Risk Factor Identification (prioritized list)
  - Recommendations (prioritized by estimated impact)
  - Anomaly Detections (with severity and timeframe)
  - Performance Forecasts (30-day projection)

## Performance Metrics

| Metric | Overall | Facebook | Google | Amazon | TikTok |
|--------|---------|----------|--------|--------|--------|
| Health Score RMSE | 7.2 | 6.8 | 7.0 | 8.1 | 9.3 |
| Classification Accuracy | 0.83 | 0.85 | 0.84 | 0.78 | 0.75 |
| Anomaly Detection F1 | 0.76 | 0.79 | 0.78 | 0.72 | 0.68 |
| Recommendation Precision@3 | 0.81 | 0.84 | 0.82 | 0.77 | 0.74 |
| Processing Time | 2.5s | 2.3s | 2.4s | 2.8s | 3.1s |

### Evaluation Methodologies
- Time series cross-validation for forecasting components
- Human expert validation for anomaly detection
- A/B testing of recommendations in production
- Confusion matrix analysis for classification
- Precision-recall analysis for recommendation relevance

## Limitations and Biases

### Known Limitations
- Requires minimum of 30 days of historical data for reliable predictions
- Performance varies with account size (better for accounts with higher spend)
- Platform changes may affect prediction accuracy until model is retrained
- Limited ability to account for external factors (market changes, seasonality)
- Recommendations based on historical patterns may not apply to novel situations

### Potential Biases
- May favor conventional account structures seen in training data
- Better performance for e-commerce and retail than B2B or services
- Can be sensitive to rapid changes in platform algorithms
- May overweight recent performance vs. long-term patterns
- Potential bias toward larger accounts with more data granularity

### Evaluation by Segment

| Segment | Health Score RMSE | Classification Accuracy | Notes |
|---------|-------------------|-------------------------|-------|
| E-commerce | 6.4 | 0.87 | Strong performance across platforms |
| B2B Services | 9.1 | 0.76 | Limited training data affects accuracy |
| Retail | 6.7 | 0.85 | Strong seasonal pattern detection |
| Finance | 8.5 | 0.79 | Regulatory constraints affect recommendations |
| Small Accounts | 9.8 | 0.74 | Less data leads to lower confidence |
| Large Accounts | 6.1 | 0.88 | More data improves prediction quality |

## Ethical Considerations

### Data Privacy
- All account data is anonymized before training and analysis
- No personally identifiable information is used or exposed
- Aggregated metrics protect individual account confidentiality
- Access controls restrict data visibility to authorized users only

### Fairness Assessment
- Regular performance audits across industry verticals
- Testing for systematic biases in recommendations
- Periodic retraining to incorporate diverse account types
- Human review of model outputs to detect unintended biases

### Potential Risks
- Over-reliance on automated recommendations without human judgment
- Potential reinforcement of platform-specific optimization strategies
- Risk of recommending short-term gains over long-term account health
- Possible reduced experimentation if recommendations are followed too rigidly

## Usage Instructions

### Required Environment
- Python 3.9+ with PyTorch 1.10+, scikit-learn 1.0+, and Prophet
- 8GB RAM minimum, 16GB recommended
- GPU beneficial but not required
- Docker container available with all dependencies

### Setup
```bash
# Install from PyPI
pip install within-account-health-predictor

# Or use Docker
docker pull within/account-health-predictor:1.5.0
docker run -p 8001:8001 within/account-health-predictor:1.5.0
```

### Inference Example
```python
from within.models import AccountHealthPredictor

# Initialize predictor
predictor = AccountHealthPredictor(version="1.5.0")

# Prepare account data
account_data = {
    "account_id": "123456789",
    "platform": "google",
    "daily_metrics": daily_performance_data,  # Last 90 days of performance data
    "account_structure": {
        "campaigns": campaign_data,
        "settings": account_settings
    }
}

# Generate health assessment
assessment = predictor.assess(account_data)

# Access results
health_score = assessment["health_score"]
health_category = assessment["health_category"]
confidence = assessment["confidence"]
risk_factors = assessment["risk_factors"]
recommendations = assessment["recommendations"]
forecast = assessment["forecast"]
```

### API Reference
Full API documentation is available at:
- [Python SDK Documentation](../../api/python_sdk.md)
- [REST API Documentation](../../api/endpoints.md#account-health)

## Maintenance

### Owner
ML Engineering Team (ml-engineering@within.co)

### Update Frequency
- Major version updates: Quarterly
- Minor improvements: Monthly
- Monitoring: Daily

### Monitoring Plan
- Daily drift detection for platform-specific metrics
- Weekly performance evaluation against human expert assessments
- Monthly retraining evaluation
- Quarterly comprehensive review of all components

### Retraining Triggers
- Performance metrics decline below thresholds
- Significant platform changes affecting feature distribution
- New account types or verticals added to customer base
- Addition of substantial new training data (>10% increase)
- Detection of new patterns or anomalies not covered by current model

## Version History

| Version | Date | Changes | Performance Delta |
|---------|------|---------|-------------------|
| 1.0.0 | 2022-03-15 | Initial production model | Baseline |
| 1.1.0 | 2022-05-20 | Improved anomaly detection | +5.2% Anomaly F1 |
| 1.2.0 | 2022-08-12 | Enhanced recommendation engine | +7.8% Rec. Precision |
| 1.3.0 | 2022-10-30 | Added TikTok platform support | N/A (new platform) |
| 1.4.0 | 2023-01-18 | Improved forecast accuracy with LSTM | -15% Forecast RMSE |
| 1.5.0 | 2023-05-22 | Added explainability features, enhanced calibration | +4.1% Classification Accuracy |

## Supplementary Materials

- [Account Health Prediction Documentation](./account_health_prediction.md)
- [Time Series Modeling Approach](./technical/time_series_modeling.md)
- [Anomaly Detection Methodology](./technical/anomaly_detection.md)
- [Recommendation Engine Documentation](./technical/recommendation_engine.md)
- [Model Evaluation Report](./evaluation/account_health_evaluation.md) 