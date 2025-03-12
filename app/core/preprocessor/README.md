# Data Preprocessing Components

This directory contains data preprocessing components for the WITHIN ML Prediction System.

## Purpose

The preprocessing system provides utilities for:
- Transforming raw data into ML-ready formats
- Ensuring data quality and consistency
- Optimizing feature representation for models
- Managing preprocessing pipelines for training and inference
- Handling edge cases and data anomalies

## Key Components

### Data Cleaning

Components for ensuring data quality:
- Missing value detection and handling
- Outlier detection and treatment
- Duplicate removal
- Data type validation and conversion
- Text cleaning for natural language inputs

### Feature Transformation

Components for modifying feature representations:
- Feature scaling (standardization, normalization)
- Categorical encoding (one-hot, label, target encoding)
- Text vectorization (TF-IDF, embeddings)
- Numerical transformations (log, power, polynomial)
- Dimensionality reduction techniques

### Feature Engineering

Components for creating new features:
- Automated feature generation
- Feature crossing and interaction
- Time-based feature extraction
- Domain-specific feature creation
- Feature selection based on importance

### Preprocessing Pipelines

Components for managing preprocessing flows:
- Pipeline construction and serialization
- Pipeline versioning and tracking
- Fit/transform separation for training vs. inference
- Parallel and distributed preprocessing
- Incremental preprocessing for streaming data

## Usage Example

```python
from app.core.preprocessor import PreprocessingPipeline, transformers

# Create preprocessing pipeline
pipeline = PreprocessingPipeline(
    steps=[
        ("missing_handler", transformers.MissingValueHandler(strategy="mean")),
        ("scaler", transformers.StandardScaler(with_mean=True, with_std=True)),
        ("encoder", transformers.CategoricalEncoder(
            categorical_features=["campaign_type", "platform", "audience_type"],
            strategy="one-hot"
        )),
        ("feature_selector", transformers.FeatureSelector(
            top_k=20,
            selection_method="mutual_info"
        ))
    ]
)

# Fit the pipeline on training data
pipeline.fit(X_train, y_train)

# Transform both training and validation data
X_train_processed = pipeline.transform(X_train)
X_valid_processed = pipeline.transform(X_valid)

# Save the pipeline
pipeline.save("models/preprocessing/ad_score_pipeline.pkl")

# Later, load the pipeline for inference
loaded_pipeline = PreprocessingPipeline.load("models/preprocessing/ad_score_pipeline.pkl")
```

## Integration Points

- **ETL Pipeline**: Connects data ingestion to preprocessing
- **ML Training**: Provides processed features for model training
- **Inference Service**: Ensures consistent preprocessing during inference
- **Data Validation**: Works with validation components to ensure data quality

## Dependencies

- NumPy and pandas for data manipulation
- scikit-learn compatible transformers
- Text processing libraries for text features
- Statistical packages for advanced transformations
- Serialization utilities for pipeline persistence 