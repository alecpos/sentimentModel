# Kaggle Dataset Integration Pipeline for WITHIN

This module provides a robust pipeline for integrating Kaggle datasets into the WITHIN Ad Score & Account Health Predictor system. It handles data downloading, validation, preprocessing, harmonization, and fairness evaluation to ensure consistent, high-quality inputs for model training and evaluation.

## Overview

The Kaggle Dataset Integration Pipeline is designed to standardize the process of incorporating external datasets into the WITHIN system, with a particular focus on fairness and regulatory compliance. It implements the data integration strategies outlined in the project requirements:

1. **Feature Harmonization**: Creates a unified feature engineering pipeline that standardizes features across different datasets.
2. **Cross-Dataset Validation**: Implements validation strategies that test models trained on one dataset against others.
3. **Synthetic Data Augmentation**: Supports custom preprocessing functions for augmenting training data.
4. **Incremental Learning**: Enables caching of processed datasets to support continuous learning.
5. **Fairness Evaluation**: Automatically evaluates fairness metrics across protected attributes.

## Core Components

### KaggleDatasetPipeline

The main class that orchestrates the entire data integration process:

```python
pipeline = KaggleDatasetPipeline(
    data_dir="data/kaggle",
    cache_dir="data/cache",
    validate_fairness=True
)

# Get predefined dataset configurations
configs = pipeline.get_dataset_configs()

# Process a specific dataset
processed_dataset = pipeline.process_dataset(configs["customer_conversion"])
```

### DatasetConfig

Configuration class for defining dataset specifications:

```python
config = DatasetConfig(
    dataset_slug="username/dataset-name",
    category=DatasetCategory.CUSTOMER_CONVERSION,
    target_column="converted",
    feature_columns=["age", "gender", "location", "time_spent"],
    protected_attributes=["age", "gender", "location"],
    schema_path=Path("app/core/schemas/my_schema.json"),
    preprocessing_fn=my_custom_preprocessing_function
)
```

### ProcessedDataset

A processed dataset ready for model training:

```python
# Access the processed data
X_train = processed_dataset.X_train
y_train = processed_dataset.y_train
X_val = processed_dataset.X_val
y_val = processed_dataset.y_val
X_test = processed_dataset.X_test
y_test = processed_dataset.y_test

# Access metadata and fairness metrics
metadata = processed_dataset.metadata
fairness_metrics = metadata.fairness_metrics

# Access protected attributes
protected_attrs = processed_dataset.protected_attributes
```

## Supported Datasets

The pipeline comes with predefined configurations for the following Kaggle datasets:

1. **Sentiment140** (`kazanova/sentiment140`): A dataset of 1.6 million tweets with sentiment labels for training the Ad Sentiment Analyzer.

2. **Social Media Sentiments** (`kashishparmar02/social-media-sentiments-analysis-dataset`): Multi-source sentiment data for enhancing the Ad Sentiment Analyzer.

3. **Customer Conversion** (`muhammadshahidazeem/customer-conversion-dataset-for-stuffmart-com`): A synthetic dataset of customer conversion events with demographic attributes for training the Ad Score Predictor.

4. **CTR Optimization** (`rahulchavan99/the-click-through-rate-ctr-optimization`): Click-through rate prediction data for the Ad Score Predictor.

## Fairness Integration

The pipeline integrates with the WITHIN fairness framework to automatically evaluate and mitigate bias:

1. **Fairness Evaluation**: Automatically calculates fairness metrics for protected attributes.
2. **Intersectional Analysis**: Evaluates combinations of protected attributes to identify nuanced biases.
3. **Visualization Generation**: Creates visualizations to aid in fairness analysis.
4. **Bias Mitigation**: Supports integration with bias mitigation techniques.

Example fairness evaluation:

```python
from app.core.fairness import FairnessEvaluator

# Initialize fairness evaluator
evaluator = FairnessEvaluator(output_dir="fairness_results")

# Evaluate fairness for a protected attribute
results = evaluator.evaluate(
    df=data_frame,
    protected_attribute="gender",
    target_column="converted",
    prediction_column="prediction"
)

# Access fairness metrics
overall_metrics = results.overall_metrics
group_metrics = results.group_metrics
```

## Data Validation

The pipeline performs extensive validation of datasets using JSON schemas:

```json
{
  "name": "customer_conversion_schema",
  "version": "1.0.0",
  "description": "Schema for the Customer Conversion Dataset",
  "required_columns": ["converted", "age", "gender", "location"],
  "column_types": {
    "converted": "int",
    "age": "int",
    "gender": "str",
    "location": "str"
  },
  "value_constraints": {
    "converted": {
      "allowed_values": [0, 1],
      "description": "1 if customer converted, 0 otherwise"
    }
  },
  "fairness_constraints": {
    "protected_attributes": ["age", "gender", "location"],
    "fairness_metrics": ["demographic_parity", "equal_opportunity"],
    "threshold": 0.1
  }
}
```

## Example Usage

See the `app/examples/kaggle_fairness_pipeline_example.py` script for a complete example of:

1. Configuring and downloading datasets from Kaggle
2. Processing and validating datasets
3. Evaluating fairness metrics across protected attributes
4. Applying fairness mitigation techniques
5. Training models on the mitigated data
6. Generating fairness visualizations and reports

Run the example with:

```bash
python app/examples/kaggle_fairness_pipeline_example.py --dataset customer_conversion --mitigation reweighing
```

## Integration with WITHIN System

The Kaggle Dataset Integration Pipeline is designed to integrate seamlessly with the WITHIN Ad Score & Account Health Predictor components:

1. **Ad Score Predictor**: Use the `ctr_optimization` and `customer_conversion` datasets.
2. **Account Health Predictor**: Use the `customer_conversion` dataset with time-series analysis.
3. **Ad Sentiment Analyzer**: Use the `sentiment140` and `social_media_sentiments` datasets.

## Regulatory Compliance

The pipeline includes built-in features to support regulatory compliance:

1. **EU AI Act**: Automatic fairness evaluation and documentation to support regulatory requirements.
2. **NIST AI RMF**: Follows the Risk Management Framework with fairness metrics and mitigation.
3. **NYC Local Law 144**: Supports bias audit requirements with comprehensive fairness evaluation.

## Prerequisites

- Kaggle API credentials (`~/.kaggle/kaggle.json` or environment variables)
- Python 3.9+
- Required packages: pandas, numpy, scikit-learn, matplotlib, kaggle

## Implementation Details

The implementation follows WITHIN ML Backend Project Standards:

- Strict type hints throughout
- Google-style docstrings
- Comprehensive error handling
- Robust logging
- Thorough data validation
- Fairness evaluation and mitigation

The codebase uses the following key external dependencies:

- **Kaggle API**: For dataset downloading
- **pandas/numpy**: For data processing
- **scikit-learn**: For preprocessing and model training
- **matplotlib**: For visualization generation

## File Structure

```
app/core/data_integration/
├── kaggle_pipeline.py       # Main pipeline implementation
├── README.md                # This documentation file
└── __init__.py              # Package initialization

app/core/schemas/
├── sentiment140_schema.json            # Schema for Sentiment140 dataset
├── social_media_sentiments_schema.json # Schema for Social Media Sentiments
├── customer_conversion_schema.json     # Schema for Customer Conversion
└── ctr_optimization_schema.json        # Schema for CTR Optimization

app/core/
├── validation.py            # Data validation implementation
├── fairness.py              # Fairness evaluation implementation
└── __init__.py              # Package initialization

app/examples/
└── kaggle_fairness_pipeline_example.py # Example implementation
```

## Future Enhancements

Planned enhancements for the pipeline include:

1. Enhanced synthetic data generation using GANs
2. Support for incremental learning with dataset versioning
3. Integration with additional fairness mitigation techniques
4. Support for multi-modal data integration (text, images, etc.)
5. Enhanced cross-dataset validation and transfer learning capabilities

## Contributing

When enhancing the Kaggle Dataset Integration Pipeline, please ensure:

1. All new features include comprehensive type annotations
2. Documentation is updated to reflect changes
3. New components include appropriate tests
4. Fairness considerations are addressed for any data processing functions
5. Any additional dependencies are justified and approved 