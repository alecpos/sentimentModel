# ML Prediction Models

DOCUMENTATION STATUS: COMPLETE

This directory contains machine learning prediction models for the WITHIN ML Prediction System.

## Purpose

The prediction module provides capabilities for:
- Ad performance prediction with multi-modal features
- Account health assessment and risk prediction
- Anomaly detection for advertising performance
- Model calibration and uncertainty quantification
- Privacy-preserving training methodologies
- Specialized neural network architectures

## Directory Structure

- **__init__.py**: Module initialization with component exports
- **base.py**: Base interfaces and abstract classes for all ML models
- **ad_score_predictor.py**: Ad performance prediction models and components
- **account_health_predictor.py**: Account health assessment models
- **anomaly_detector.py**: Anomaly detection models for performance outliers
- **training.py**: Training utilities, validation, and model persistence
- **example_usage.py**: Example usage patterns and code snippets

## Key Components

### BaseMLModel

`BaseMLModel` is responsible for providing a standardized interface for all machine learning models in the system.

**Key Features:**
- Defines common interface for prediction models
- Provides abstract methods for training and inference
- Implements serialization/deserialization functionality
- Includes metrics calculation and validation utilities

**Parameters:**
- `model_name` (str): Unique identifier for the model
- `feature_names` (List[str]): Names of features used by the model
- `model_version` (str, optional): Model version identifier
- `metadata` (Dict, optional): Additional model metadata

**Methods:**
- `predict(X)`: Makes predictions for input features X
- `fit(X, y)`: Trains the model on provided data
- `save(path)`: Serializes and saves the model to disk
- `load(path)`: Loads a serialized model from disk
- `validate(X, y)`: Validates the model on test data

### AdScorePredictor

`AdScorePredictor` is responsible for predicting ad performance scores based on multi-modal features.

**Key Features:**
- Multi-modal feature processing (text, image, user behavior)
- Calibrated prediction intervals
- Model explainability via SHAP values
- Fair predictions across demographic groups
- Hierarchical modeling for account-level effects

**Parameters:**
- `feature_config` (Dict): Configuration for feature processing
- `model_type` (str): Type of model to use ('ensemble', 'nn', 'linear')
- `calibration` (bool): Whether to apply calibration
- `explainable` (bool): Whether to enable explainability features
- `uncertainty` (bool): Whether to quantify prediction uncertainty

**Methods:**
- `predict(ad_features)`: Predicts performance score for an ad
- `predict_with_intervals(ad_features)`: Predicts with confidence intervals
- `explain(ad_features)`: Provides feature importance explanation
- `update(new_data)`: Updates the model with new training data
- `get_feature_importance()`: Returns global feature importance

### AdPredictorNN

`AdPredictorNN` is responsible for neural network-based ad performance prediction with advanced architectural features.

**Key Features:**
- Multi-headed attention for feature interaction
- Cross-modal fusion of text and image features
- Adaptive regularization based on data characteristics
- Quantized inference for production deployment
- Support for mixed precision training

**Parameters:**
- `input_dim` (int): Dimension of input features
- `hidden_dims` (List[int]): Dimensions of hidden layers
- `dropout_rate` (float): Rate for dropout regularization
- `enable_quantum_noise` (bool): Whether to use quantum noise layers
- `use_cross_modal` (bool): Whether to use cross-modal attention

**Methods:**
- `forward(X)`: Forward pass through the neural network
- `feature_extraction(X)`: Extracts intermediate features
- `attention_weights(X)`: Returns attention weights for interpretability
- `quantize()`: Quantizes the model for efficient inference
- `fuse_bn()`: Fuses batch normalization for inference

### CalibratedEnsemble

`CalibratedEnsemble` is responsible for combining multiple models with calibration to improve prediction accuracy and reliability.

**Key Features:**
- Ensemble of diverse base models
- Temperature scaling for probability calibration
- Weighted model averaging based on recent performance
- Outlier-robust aggregation methods
- Uncertainty quantification via ensemble variance

**Parameters:**
- `base_models` (List[BaseMLModel]): List of models to ensemble
- `weights` (List[float], optional): Weighting for each model
- `calibration_method` (str): Method for calibrating probabilities
- `aggregation_method` (str): Method for aggregating predictions

**Methods:**
- `predict(X)`: Makes ensemble predictions
- `calibrate(X_val, y_val)`: Calibrates the ensemble on validation data
- `add_model(model, weight)`: Adds a new model to the ensemble
- `remove_model(model_idx)`: Removes a model from the ensemble
- `get_model_weights()`: Returns the current model weights

### DynamicLinear

`DynamicLinear` is responsible for implementing adaptive linear models with dynamic feature selection.

**Key Features:**
- Online feature selection and importance updating
- Adaptive L1/L2 regularization based on data
- Time-decay weighting for temporal data
- Efficient sparse matrix operations
- Interpretable coefficient tracking

**Parameters:**
- `n_features` (int): Number of input features
- `regularization` (str): Regularization type ('l1', 'l2', 'elastic')
- `alpha` (float): Regularization strength
- `time_decay` (float): Decay factor for temporal weighting

**Methods:**
- `fit_incremental(X, y)`: Incrementally updates the model
- `predict(X)`: Makes predictions with the current model
- `get_coefficients()`: Returns model coefficients
- `sparsify()`: Applies sparsification to remove near-zero weights
- `reset_features(indices)`: Resets weights for specified features

### AdaptiveDropout

`AdaptiveDropout` is responsible for implementing a dropout layer with dynamic rate adjustment based on model uncertainty.

**Key Features:**
- Adaptively adjusts dropout rate during training
- Uses recent gradient information to tune regularization
- Applies different rates to different feature types
- Reduces overfitting while preserving important features
- Improves generalization on heterogeneous data

**Parameters:**
- `p` (float): Initial dropout probability
- `momentum` (float): Momentum factor for rate adaptation
- `window_size` (int): Window size for gradient history
- `adaptive_range` (Tuple[float, float]): Min/max dropout range

**Methods:**
- `forward(x)`: Applies dropout with current rate
- `update_rate(grads)`: Updates dropout rate based on gradients
- `reset()`: Resets dropout rate to initial value
- `get_current_rate()`: Returns current dropout rate

### HierarchicalCalibrator

`HierarchicalCalibrator` is responsible for calibrating predictions across different hierarchical levels (campaign, ad group, ad).

**Key Features:**
- Accounts for hierarchical structure in advertising data
- Applies different calibration at each hierarchical level
- Preserves relative ordering within groups
- Handles sparse data at lower hierarchy levels
- Shares statistical strength across the hierarchy

**Parameters:**
- `hierarchy_levels` (List[str]): Names of hierarchical levels
- `calibration_method` (str): Method for calibration
- `smoothing` (float): Smoothing parameter for sparse levels
- `max_iter` (int): Maximum iterations for optimization

**Methods:**
- `fit(X, y, groups)`: Fits calibrators at each hierarchical level
- `transform(X, groups)`: Applies calibration hierarchically
- `fit_transform(X, y, groups)`: Fits and applies calibration
- `get_calibration_mapping()`: Returns calibration functions by level

### GeospatialCalibrator

`GeospatialCalibrator` is responsible for calibrating predictions based on geospatial factors and regional performance variations.

**Key Features:**
- Region-specific calibration curves
- Spatial smoothing for similar regions
- Handles geographic hierarchies (country, state, city)
- Accounts for population density effects
- Seasonal adjustment by region

**Parameters:**
- `geo_level` (str): Level of geographic aggregation
- `smoothing_neighbors` (int): Number of neighbors for smoothing
- `min_samples_per_region` (int): Minimum samples required per region
- `fallback_strategy` (str): Strategy when region has insufficient data

**Methods:**
- `fit(X, y, regions)`: Fits calibration by geographic region
- `transform(X, regions)`: Calibrates predictions for specified regions
- `get_region_factors()`: Returns calibration factors by region
- `plot_region_map()`: Visualizes calibration factors geographically

### PerformanceMonitor

`PerformanceMonitor` is responsible for tracking model performance metrics over time and detecting performance degradation.

**Key Features:**
- Real-time tracking of prediction accuracy
- Drift detection in feature distributions
- Performance comparison against baselines
- Automated alerting for degradation
- Detailed performance breakdown by segments

**Parameters:**
- `metrics` (List[str]): Performance metrics to monitor
- `window_size` (int): Window size for calculating metrics
- `alert_threshold` (float): Threshold for triggering alerts
- `baseline_model` (BaseMLModel, optional): Model to compare against

**Methods:**
- `log_prediction(prediction, actual)`: Logs a prediction and actual value
- `get_current_metrics()`: Returns current performance metrics
- `detect_drift()`: Checks for performance drift
- `generate_report()`: Generates a detailed performance report
- `reset()`: Resets the monitoring state

### CrossModalAttention

`CrossModalAttention` is responsible for implementing attention mechanisms that fuse features across different modalities.

**Key Features:**
- Attention-based fusion of text, image, and numeric features
- Modality-specific encoding layers
- Multi-head attention for complex interactions
- Learnable modality importance weights
- Support for missing modalities

**Parameters:**
- `modality_dims` (Dict[str, int]): Dimensions for each modality
- `attention_heads` (int): Number of attention heads
- `attention_dim` (int): Dimension of attention layers
- `dropout` (float): Dropout rate for attention

**Methods:**
- `forward(modalities)`: Processes and fuses multiple modalities
- `get_attention_weights()`: Returns attention weights for interpretability
- `encode_modality(modality, data)`: Encodes a specific modality

### MultiModalFeatureExtractor

`MultiModalFeatureExtractor` is responsible for extracting and processing features from multiple data modalities.

**Key Features:**
- Processes text using transformer models
- Extracts visual features using CNN architectures
- Handles tabular data with specialized preprocessing
- Performs feature selection for each modality
- Aligns features across modalities for fusion

**Parameters:**
- `text_model` (str): Name of text feature extractor
- `image_model` (str): Name of image feature extractor
- `tabular_preprocessor` (Dict): Configuration for tabular features
- `output_dim` (int): Dimension of final features

**Methods:**
- `extract_features(data)`: Extracts features from multimodal data
- `process_text(text)`: Processes text modality
- `process_image(image)`: Processes image modality
- `process_tabular(tabular)`: Processes tabular data

### QuantumNoiseLayer

`QuantumNoiseLayer` is responsible for adding quantum-inspired noise for regularization and optimization improvements.

**Key Features:**
- Quantum-inspired noise generation
- Helps escape local minima during training
- Improves model robustness to adversarial examples
- Configurable noise distribution and strength
- Annealing schedule for noise reduction

**Parameters:**
- `channels` (int): Number of feature channels
- `noise_type` (str): Type of quantum noise to apply
- `noise_intensity` (float): Intensity of the noise
- `annealing_schedule` (str): Schedule for reducing noise

**Methods:**
- `forward(x)`: Applies quantum noise to input
- `update_schedule(step)`: Updates noise based on training step
- `get_current_noise_level()`: Returns current noise intensity

### SplineCalibrator

`SplineCalibrator` is responsible for calibrating model outputs using flexible spline-based transformations.

**Key Features:**
- Non-parametric calibration using splines
- Preserves prediction ordering while improving calibration
- Handles multi-class calibration
- Supports both piecewise and smoothed splines
- More flexible than traditional methods like Platt scaling

**Parameters:**
- `n_knots` (int): Number of spline knots
- `knot_spacing` (str): Method for spacing knots
- `spline_order` (int): Order of the spline
- `regularization` (float): Regularization strength

**Methods:**
- `fit(y_pred, y_true)`: Fits the calibration spline
- `transform(y_pred)`: Applies calibration transformation
- `get_spline_parameters()`: Returns spline parameters
- `plot_calibration_curve()`: Visualizes calibration mapping

### DPTrainingValidator

`DPTrainingValidator` is responsible for validating that model training adheres to differential privacy guarantees.

**Key Features:**
- Validates adherence to DP-SGD training requirements
- Tracks privacy budget consumption
- Verifies clipping bounds and noise multipliers
- Performs privacy auditing on trained models
- Provides certification of privacy guarantees

**Parameters:**
- `epsilon` (float): Privacy budget
- `delta` (float): Failure probability
- `max_grad_norm` (float): Gradient clipping norm
- `noise_multiplier` (float): Noise multiplier for DP-SGD

**Methods:**
- `validate_training_params(params)`: Validates training parameters
- `track_privacy_budget(steps)`: Tracks privacy budget consumption
- `audit_model(model)`: Audits a model for privacy leakage
- `generate_privacy_report()`: Generates a privacy guarantee report

### AccountHealthPredictor

`AccountHealthPredictor` is responsible for predicting the overall health and risk level of advertising accounts.

**Key Features:**
- Combines multiple health indicators into a unified score
- Identifies accounts at risk of performance degradation
- Predicts churn probability for accounts
- Provides actionable insights for account improvement
- Early warning system for account issues

**Parameters:**
- `indicators` (List[str]): Health indicators to track
- `risk_thresholds` (Dict[str, float]): Thresholds for risk levels
- `time_window` (int): Window size for temporal analysis
- `model_type` (str): Underlying model type

**Methods:**
- `predict_health(account_data)`: Predicts overall account health
- `predict_churn_risk(account_data)`: Predicts risk of account churn
- `get_contributing_factors(account_id)`: Identifies factors affecting health
- `recommend_actions(account_id)`: Recommends actions to improve health

### AnomalyDetector

`AnomalyDetector` is responsible for detecting anomalies in advertising performance data.

**Key Features:**
- Multi-dimensional anomaly detection
- Accounts for seasonal and temporal patterns
- Adaptive thresholds based on account history
- Differentiates between positive and negative anomalies
- Root cause analysis for detected anomalies

**Parameters:**
- `detection_method` (str): Method for anomaly detection
- `sensitivity` (float): Sensitivity of detection
- `dimensions` (List[str]): Dimensions to monitor for anomalies
- `baseline_window` (int): Window size for establishing baseline

**Methods:**
- `detect_anomalies(performance_data)`: Detects anomalies in performance data
- `update_baseline(new_data)`: Updates the baseline with new data
- `get_anomaly_score(data_point)`: Calculates anomaly score for a data point
- `explain_anomaly(anomaly_id)`: Provides explanation for a detected anomaly

## Usage Examples

### AdScorePredictor Usage

```python
from app.models.ml.prediction import AdScorePredictor

# Initialize the predictor with configuration
predictor = AdScorePredictor(
    feature_config={
        "text_features": ["headline", "description"],
        "image_features": ["main_image"],
        "numerical_features": ["budget", "bid"]
    },
    model_type="ensemble",
    calibration=True,
    explainable=True
)

# Prepare ad features
ad_features = {
    "headline": "Limited time offer on premium products",
    "description": "Shop now for exclusive deals on our top-rated items",
    "main_image": image_tensor,
    "budget": 500.0,
    "bid": 0.75
}

# Get ad score prediction
score = predictor.predict(ad_features)

# Get prediction with confidence interval
score, lower_bound, upper_bound = predictor.predict_with_intervals(ad_features)

# Get feature importance explanation
explanation = predictor.explain(ad_features)
```

### AnomalyDetector Usage

```python
from app.models.ml.prediction import AnomalyDetector
import pandas as pd

# Initialize detector
detector = AnomalyDetector(
    detection_method="isolation_forest",
    sensitivity=0.95,
    dimensions=["clicks", "conversions", "ctr", "cvr"],
    baseline_window=30
)

# Load performance data
performance_data = pd.read_csv("ad_performance.csv")

# Detect anomalies
anomalies = detector.detect_anomalies(performance_data)

# Get explanation for a specific anomaly
anomaly_id = anomalies[0]["id"]
explanation = detector.explain_anomaly(anomaly_id)

print(f"Found {len(anomalies)} anomalies")
print(f"Explanation for anomaly #{anomaly_id}: {explanation}")
```

## Integration Points

- **Prediction API**: Provides endpoints for real-time predictions
- **Training Pipeline**: Integrates with model training and evaluation
- **Monitoring System**: Supplies prediction data for monitoring
- **Dashboard**: Provides visualizations and explanations for predictions
- **Alert System**: Triggers alerts for anomalies and performance issues
- **AutoML**: Interfaces with automated feature selection and tuning

## Dependencies

- **PyTorch**: Deep learning framework for neural networks
- **scikit-learn**: Machine learning utilities and algorithms
- **NumPy/Pandas**: Data manipulation and preprocessing
- **SHAP**: Model explainability and feature importance
- **PyOD**: Outlier detection algorithms
- **Opacus**: Differential privacy for PyTorch
- **Transformers**: Text processing for multi-modal features
