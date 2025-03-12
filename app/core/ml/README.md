# Core ML Components

This directory contains core machine learning utilities for the WITHIN ML Prediction System.

DOCUMENTATION STATUS: COMPLETE

## Purpose

The core ML system provides low-level capabilities for:
- Loading and managing ML models
- Processing and transforming features
- Handling predictions and outputs
- Implementing reusable ML pipeline components
- Providing common ML utility functions

## Key Components

### Model Management

Utilities for working with ML models:
- Model loading and initialization
- Model versioning and selection
- Model metadata handling
- Format conversion and compatibility
- Memory management for models

### Feature Processing

Utilities for handling model features:
- Feature normalization and scaling
- Feature transformation pipelines
- Type conversion and validation
- Handling missing values
- Feature encoding

### Prediction Handling

Utilities for working with model predictions:
- Prediction formatting and standardization
- Confidence score calculation
- Threshold application
- Prediction logging and tracking
- Output validation

## Usage Example

```python
from app.core.ml import load_model, normalize_features, format_prediction

# Load a trained model
model = load_model("models/ad_score_model_v1.pkl")

# Prepare features for prediction
input_features = {
    "campaign_id": "C123456",
    "budget": 1500.0,
    "creative_quality": 0.85,
    "target_audience": "high_intent",
    "platform": "facebook"
}

# Normalize features for the model
normalized_features = normalize_features(input_features)

# Generate prediction
raw_prediction = model.predict(normalized_features)

# Format the prediction for API response
formatted_prediction = format_prediction(raw_prediction)
print(formatted_prediction)
# Output: {"prediction": "0.87", "confidence": 0.92, "version": "1.0"}
```

## Integration Points

- **ML Models**: Specific model implementations use these core utilities
- **API Endpoints**: Prediction endpoints leverage these utilities
- **Feature Store**: Feature processing uses these core components
- **Model Monitoring**: The monitoring system uses prediction formatting
- **ML Pipeline**: The ML pipeline is built on these core functions

## Dependencies

- Python standard libraries (typing, logging)
- ML framework utilities (abstracted to support multiple frameworks)
- Core error handling components
- Configuration system for ML parameters 