# Fairness Evaluation Module

## Overview

The Fairness Evaluation Module provides comprehensive tools for assessing and mitigating bias in machine learning models within the WITHIN Ad Score & Account Health Predictor system. This module enables the evaluation of fairness metrics across different demographic groups, visualization of fairness results, and implementation of bias mitigation techniques.

## Key Components

### 1. Fairness Metrics

The module supports various fairness metrics, including:

- **Demographic Parity**: Ensures similar prediction rates across demographic groups
- **Equal Opportunity**: Ensures similar true positive rates across demographic groups
- **Predictive Parity**: Ensures similar precision across demographic groups
- **Accuracy Parity**: Ensures similar accuracy across demographic groups
- **False Positive Rate Parity**: Ensures similar false positive rates across demographic groups
- **False Negative Rate Parity**: Ensures similar false negative rates across demographic groups
- **Treatment Equality**: Ensures similar ratios of false negatives to false positives across groups
- **Disparate Impact**: Measures the ratio of positive prediction rates between groups

### 2. Fairness Evaluator

The `FairnessEvaluator` class is the core component for assessing fairness in models:

```python
from app.core.fairness import FairnessEvaluator, FairnessThreshold

# Initialize the evaluator
evaluator = FairnessEvaluator(
    output_dir="./fairness_results",
    threshold=0.1,  # Standard fairness threshold
    save_visualizations=True
)

# Evaluate fairness for a protected attribute
results = evaluator.evaluate(
    df=evaluation_data,
    protected_attribute="gender",
    target_column="target",
    prediction_column="prediction"
)

# Print summary
print(results.summary())
```

### 3. Fairness Mitigation Techniques

The module provides several bias mitigation techniques:

#### Reweighing

Mitigates bias by assigning different weights to training examples:

```python
from app.core.fairness import Reweighing

# Initialize reweigher
reweigher = Reweighing(protected_attribute="gender")

# Fit to training data
reweigher.fit(X_train, y_train)

# Get sample weights for training
sample_weights = reweigher.get_sample_weights(X_train, y_train)

# Train model with weights
model.fit(X_train, y_train, sample_weight=sample_weights)
```

#### Fair Data Transformer

Transforms features to remove correlation with protected attributes:

```python
from app.core.fairness import FairDataTransformer

# Initialize transformer
transformer = FairDataTransformer(
    protected_attribute="gender",
    method="decorrelation"
)

# Transform data
X_transformed = transformer.fit_transform(X_train, y_train)

# Train model on transformed data
model.fit(X_transformed, y_train)
```

### 4. Visualization Capabilities

The module generates various visualizations to help interpret fairness results:

- **Fairness Metrics Bar Charts**: Visual comparison of fairness metrics against thresholds
- **Group Metrics Comparison**: Comparison of performance metrics across demographic groups
- **Protected Attribute Distribution**: Distribution of outcomes by protected attribute
- **Intersectional Heatmaps**: Visualization of metrics across combinations of protected attributes

## Integration with Kaggle Dataset Pipeline

The fairness module integrates seamlessly with the Kaggle dataset pipeline:

```python
from app.core.data_integration.kaggle_pipeline import KaggleDatasetPipeline

# Initialize pipeline with fairness validation
pipeline = KaggleDatasetPipeline(
    data_dir="./data",
    cache_dir="./cache",
    validate_fairness=True
)

# Process dataset with fairness evaluation
processed_dataset = pipeline.process_dataset(dataset_config)
```

## Example Usage

See the example scripts for complete demonstrations:

- `app/examples/sentiment_fairness_example_local.py`: Local example with synthetic data
- `app/examples/sentiment140_fairness_example.py`: Example using the Kaggle Sentiment140 dataset

## Fairness Thresholds

The module supports different fairness thresholds:

- **STRICT**: 0.05 (5% difference allowed)
- **STANDARD**: 0.1 (10% difference allowed)
- **RELAXED**: 0.2 (20% difference allowed)

## Requirements

- Python 3.9+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Future Enhancements

- Additional fairness metrics (e.g., equalized odds)
- More mitigation techniques (e.g., adversarial debiasing)
- Interactive fairness dashboards
- Automated fairness monitoring for production models 