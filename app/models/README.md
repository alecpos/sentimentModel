# Models Directory

This directory contains data and ML models for the WITHIN ML Prediction System. It's organized into several key subdirectories that handle different aspects of data modeling and machine learning.

DOCUMENTATION STATUS: COMPLETE

## Purpose

The models system provides capabilities for:
- Defining database schema and data persistence
- Implementing ML models for prediction and analysis
- Representing domain-specific business entities
- Validating and monitoring model performance
- Supporting fairness and robustness in ML systems

## Directory Structure

### ml/
Machine learning models and components for prediction, monitoring, and evaluation.

- **prediction/**: Core prediction models (Ad Score, Account Health, Anomaly Detection)
- **monitoring/**: Model monitoring and drift detection
- **validation/**: Model validation tools and A/B testing framework
- **fairness/**: Fairness assessment and bias mitigation components
- **robustness/**: Robustness testing and model attack defenses

### database/
SQLAlchemy database models for persistence and relational data storage, organized by domain:
- **users/**: User management and authentication models
- **campaigns/**: Campaign and ad performance models
- **ml_system/**: ML operations and tracking models
- **analytics/**: Analytics and reporting models

### domain/
Domain entities and business logic models that represent core business concepts:
- Data Lake models for data storage and access
- Data Catalog models for metadata management
- Business entity definitions and relationships

## Key Components

### MODEL_REGISTRY
`MODEL_REGISTRY` is a centralized configuration dictionary that stores metadata about trained models in the system, including:
- Current model versions
- Last training dates
- Performance metrics

This registry enables version control, reproducibility, and audit trails for all production models.

### Submodules

#### ml
`ml` module contains machine learning models, algorithms, and utilities for:
- Prediction systems (ad scoring, account health)
- Anomaly detection
- Model monitoring and drift detection
- Model validation and A/B testing
- Fairness assessments and bias mitigation
- Robustness testing and model attack defenses

Access to key ML components is provided through factory functions and direct imports:

```python
from app.models import ml
from app.models.ml import BaseMLModel, ModelEvaluator
```

#### database
`database` module provides SQLAlchemy models for data persistence, including:
- User management and authentication models
- Campaign and ad performance tracking
- ML operations metrics and storage
- Analytics and reporting data

Database models can be accessed through direct imports:

```python
from app.models import database
from app.models.database.users import UserModel
```

#### domain
`domain` module contains domain entities and business logic models:
- Data Lake models for data storage access
- Data Catalog models for metadata management
- Business entity definitions
- Value objects and domain primitives

Domain models can be accessed through direct imports:

```python
from app.models import domain
from app.models.domain import DataLakeModel
```

### Database Models

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

## Usage Examples

```python
# Working with ML models
from app.models.ml import get_ad_score_predictor
from app.models.ml.fairness import FairnessEvaluator

# Initialize predictor
predictor_class = get_ad_score_predictor()
predictor = predictor_class()
predictor.fit(training_data, training_labels)

# Make predictions
ad_scores = predictor.predict(new_ads_data)

# Working with database models
from app.models import AdScoreModel
from app.core.db import get_db

# Create a new ad score record
db = next(get_db())
new_score = AdScoreModel(
    ad_id="ad123",
    advertiser_id="adv456",
    ad_content="Experience our new product today!",
    engagement_score=0.87,
    sentiment_score=0.92,
    complexity_score=0.34,
    topics=["product", "promotion"],
    target_audience_match=0.78,
    predicted_ctr=0.043,
    confidence_score=0.89
)
db.add(new_score)
db.commit()

# Access model registry information
from app.models import MODEL_REGISTRY

ad_score_version = MODEL_REGISTRY["ad_score"]["version"]
last_trained = MODEL_REGISTRY["ad_score"]["last_trained"]
f1_score = MODEL_REGISTRY["ad_score"]["metrics"]["f1"]

print(f"Using Ad Score model version {ad_score_version}, last trained on {last_trained}")
print(f"Model F1 score: {f1_score}")

# Working with domain models
from app.models.domain import DataLakeModel

data_lake_entry = DataLakeModel(
    name="campaign_performance_2023",
    path="s3://within-data-lake/campaigns/2023/",
    format="parquet",
    partition_keys=["date", "campaign_id"]
)

# Access the data
performance_data = data_lake_entry.get_data(
    filters={"date": "2023-09-01"}
)
```

## Integration Points

- **API Layer**: Models define data structures for API endpoints
- **ETL Pipeline**: Models guide data transformation processes
- **ML System**: Models provide prediction and analysis capabilities
- **Reporting System**: Models support data aggregation and analytics
- **Security System**: Models enforce access control and validation

## Dependencies

- **SQLAlchemy**: For ORM database models
- **PyTorch**: For neural network models
- **scikit-learn**: For traditional ML models
- **Pydantic**: For data validation and schemas
- **NumPy/Pandas**: For data manipulation
- **XGBoost**: For gradient boosting models
- **SHAP**: For model explanation

## Additional Resources

- See `ml/README.md` for detailed information about machine learning models
- See `database/README.md` for database schema documentation
- See `domain/README.md` for domain model documentation 