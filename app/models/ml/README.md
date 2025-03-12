# Machine Learning Models

DOCUMENTATION STATUS: COMPLETE

This directory contains all machine learning components for the WITHIN ML Prediction System.

## Purpose

The machine learning module provides:
- Prediction models for ad scoring and account health assessment
- Fairness evaluation and mitigation tools
- Robustness testing and model attack defenses
- Model validation and A/B testing framework
- Performance monitoring and drift detection

## Directory Structure

- **prediction/**: Core prediction models (Ad Score, Account Health, Anomaly Detection)
- **monitoring/**: Model monitoring and drift detection tools
- **validation/**: Model validation tools and A/B testing framework
- **fairness/**: Fairness assessment and bias mitigation components
- **robustness/**: Robustness testing and model attack defenses

## Key Components

### Database Models (Re-exported)

#### BaseModel
`BaseModel` is the foundational SQLAlchemy model class that all database models inherit from. It provides:
- Common columns (creation/update timestamps)
- Serialization methods
- Query utilities
- Consistent configuration across all models

#### AdScoreModel
`AdScoreModel` is a SQLAlchemy ORM model for storing ad scoring results in the database. It includes:
- Ad identity and metadata
- Multiple score components (engagement, sentiment, complexity)
- Topic classification data
- Audience matching metrics
- Prediction confidence scores

#### AdScoreAnalysisModel
`AdScoreAnalysisModel` stores detailed analysis results for ad scoring, including:
- Content feature extraction
- Linguistic analysis data
- Visual element classification
- Performance projections
- Similar successful ads for reference
- Improvement suggestions

#### AdAccountHealthModel
`AdAccountHealthModel` stores health metrics and diagnostics for advertising accounts, including:
- Overall account health score
- Engagement trends over time
- Risk factors and warnings
- Optimization suggestions
- Historical performance data
- Spend efficiency metrics
- Audience health indicators

#### PerformanceMetricModel
`PerformanceMetricModel` stores time-series performance metrics for accounts, including:
- Metric name and value
- Time period and date range
- Anomaly detection flags
- Anomaly scores for detected issues
- Metadata for contextual information

### Configuration

#### MODEL_REGISTRY
`MODEL_REGISTRY` is a centralized configuration dictionary that stores metadata about trained models in the system, including:
- Current model versions
- Last training dates
- Performance metrics

This registry enables version control, reproducibility, and audit trails for all production models.

### Factory Functions

#### get_ad_score_predictor()
Factory function that returns the `AdScorePredictor` class (not an instance) from the prediction module.
This allows for lazy loading and dependency isolation.

```python
predictor_class = get_ad_score_predictor()
predictor = predictor_class()
scores = predictor.predict(ad_data)
```

#### get_health_predictor()
Factory function that returns the `AccountHealthPredictor` class (not an instance) from the prediction module.
This enables consistent access to the predictor while allowing for version control.

```python
predictor_class = get_health_predictor()
predictor = predictor_class()
health_metrics = predictor.predict(account_data)
```

#### get_anomaly_detector()
Factory function that returns the `AnomalyDetector` class (not an instance) from the prediction module.
This provides a standardized way to access anomaly detection capabilities.

```python
detector_class = get_anomaly_detector()
detector = detector_class()
anomalies = detector.detect(performance_data)
```

### Submodules

#### prediction
The `prediction` submodule contains the core machine learning models for making predictions about ad scores, account health, and anomaly detection. It includes:
- Model definitions for various prediction tasks
- Feature preprocessing and transformation utilities
- Model training and evaluation workflows
- Inference optimization techniques
- Ensemble methods for improved prediction accuracy

See [prediction/README.md](prediction/README.md) for complete documentation.

#### monitoring
The `monitoring` submodule provides tools for monitoring model performance in production, including:
- Drift detection for data and concept drift
- Feature distribution monitoring
- Feature correlation monitoring
- Alert management for performance degradation
- Automated retraining triggers

See [monitoring/README.md](monitoring/README.md) for complete documentation.

#### validation
The `validation` submodule contains components for validating and testing machine learning models before deployment, including:
- A/B testing framework for comparing model performance
- Shadow deployment for risk-free testing in production
- Canary deployment for staged rollouts
- Golden set validation for regression testing
- Statistical significance testing for model comparisons

See [validation/README.md](validation/README.md) for complete documentation.

#### fairness
The `fairness` submodule provides tools for assessing and mitigating biases in machine learning models, including:
- Bias detection across protected groups
- Fairness metrics calculation and evaluation
- Bias mitigation techniques during training
- Fairness constraints and regularization methods
- Counterfactual generation for fairness testing

See [fairness/README.md](fairness/README.md) for complete documentation.

#### robustness
The `robustness` submodule contains components for testing and improving model robustness against various challenges, including:
- Adversarial attack simulation
- Model robustness certification
- Noise resilience evaluation
- Input perturbation testing
- Defense mechanisms against attacks

See [robustness/README.md](robustness/README.md) for complete documentation.

## Core Submodules

See individual README files in each directory for detailed documentation:

- **[prediction/README.md](prediction/README.md)**: Documentation for prediction models
- **[monitoring/README.md](monitoring/README.md)**: Documentation for monitoring tools
- **[validation/README.md](validation/README.md)**: Documentation for validation framework
- **[fairness/README.md](fairness/README.md)**: Documentation for fairness components
- **[robustness/README.md](robustness/README.md)**: Documentation for robustness testing

## Usage Examples

### Working with Predictors

```python
from app.models.ml import get_ad_score_predictor, get_health_predictor

# Initialize ad score predictor
ad_predictor_class = get_ad_score_predictor()
ad_predictor = ad_predictor_class()

# Train the model
ad_predictor.fit(ad_training_data, ad_training_labels)

# Make predictions
ad_scores = ad_predictor.predict(new_ads_data)

# Initialize health predictor
health_predictor_class = get_health_predictor()
health_predictor = health_predictor_class()

# Get account health assessment
health_metrics = health_predictor.predict(account_data)
```

### Working with Anomaly Detection

```python
from app.models.ml import get_anomaly_detector

# Initialize anomaly detector
detector_class = get_anomaly_detector()
detector = detector_class(sensitivity=0.95)

# Train detector on normal data
detector.fit(normal_performance_data)

# Detect anomalies in new data
anomalies = detector.detect(new_performance_data)
if anomalies:
    print(f"Found {len(anomalies)} anomalies")
    for anomaly in anomalies:
        print(f"Anomaly detected: {anomaly['metric']} at {anomaly['timestamp']}")
```

### Using the Model Registry

```python
from app.models.ml import MODEL_REGISTRY

# Get information about current models
for model_name, model_info in MODEL_REGISTRY.items():
    print(f"\nModel: {model_name}")
    print(f"Version: {model_info['version']}")
    print(f"Last trained: {model_info['last_trained']}")
    print("Performance metrics:")
    for metric_name, value in model_info['metrics'].items():
        print(f"  - {metric_name}: {value}")

# Check if a model needs retraining
from datetime import datetime
from dateutil.parser import parse

def needs_retraining(model_name, max_age_days=30):
    if model_name not in MODEL_REGISTRY:
        return True
        
    last_trained = parse(MODEL_REGISTRY[model_name]["last_trained"])
    age_days = (datetime.now() - last_trained).days
    return age_days > max_age_days

if needs_retraining("ad_score"):
    print("Ad score model needs retraining")
```

## Integration Points

- **Model Registry**: Central model metadata storage
- **Feature Store**: Provides consistent features for model training and inference
- **ML Pipeline**: Orchestrates model training and evaluation
- **API Gateway**: Interfaces with prediction endpoints
- **Monitoring System**: Tracks model performance and alerts on issues
- **Experiment Tracking**: Records training runs and model performance

## Dependencies

- **PyTorch**: Neural network implementation
- **scikit-learn**: Traditional ML algorithms
- **NumPy/Pandas**: Data manipulation
- **SQLAlchemy**: Database ORM
- **SHAP**: Model explanation framework
- **XGBoost**: Gradient boosting
- **AlibiDetect**: Drift detection algorithms 