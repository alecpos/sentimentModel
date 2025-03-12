# ML Model Monitoring

DOCUMENTATION STATUS: COMPLETE

This directory contains monitoring tools for machine learning models in the WITHIN ML Prediction System.

## Purpose

The monitoring module provides capabilities for:
- Data drift detection for input features
- Concept drift detection for model predictions
- Feature distribution monitoring
- Feature correlation monitoring
- Performance degradation alerting system
- Automated model retraining triggers

## Directory Structure

- **__init__.py**: Module initialization with component exports
- **drift_detector.py**: Base drift detection functionality and generic drift detectors
- **concept_drift_detector.py**: Enhanced concept drift detection with additional algorithms
- **prediction_drift_detector.py**: Specialization for monitoring model prediction patterns
- **feature_monitor.py**: Feature distribution and correlation monitoring tools
- **alert_manager.py**: Alert generation, routing, and management system

## Key Components

### DriftDetector

`DriftDetector` is responsible for providing a base class for all drift detection algorithms with common functionality.

**Key Features:**
- Abstract base class for all drift detection implementations
- Provides statistical testing framework for drift detection
- Implements windowization for streaming data
- Offers calibration utilities for threshold determination
- Includes visualization tools for drift analysis

**Parameters:**
- `reference_data` (np.ndarray): Reference distribution data
- `window_size` (int): Size of the sliding window for drift detection
- `significance_level` (float): Statistical significance level for drift tests
- `test_method` (str): Statistical test to use ('ks', 'chi2', 'wasserstein')

**Methods:**
- `detect(current_data)`: Detects if drift has occurred in current data
- `update_reference(new_reference)`: Updates the reference distribution
- `get_drift_score(current_data)`: Returns a numerical measure of drift
- `calibrate(validation_data)`: Calibrates drift thresholds using validation data
- `visualize_drift(current_data)`: Generates visualization of drift metrics

### DataDriftDetector

`DataDriftDetector` is responsible for detecting shifts in input feature distributions that could impact model performance.

**Key Features:**
- Monitors individual feature distributions for changes
- Handles both numerical and categorical features
- Provides feature-wise and global drift metrics
- Supports multidimensional drift detection
- Includes drift attribution to specific features

**Parameters:**
- `reference_data` (pd.DataFrame): Reference dataset
- `feature_columns` (List[str]): Features to monitor for drift
- `categorical_columns` (List[str], optional): Columns to treat as categorical
- `drift_threshold` (float): Threshold for detecting significant drift
- `distance_metric` (str): Distance metric for distribution comparison

**Methods:**
- `detect_drift(current_data)`: Detects if drift has occurred across features
- `get_feature_drift_scores(current_data)`: Returns drift scores per feature
- `identify_drifting_features(current_data)`: Identifies which features are drifting
- `calculate_global_drift_score(current_data)`: Calculates an aggregate drift score
- `visualize_feature_drift(feature, current_data)`: Visualizes drift for a specific feature

### ConceptDriftDetector

`ConceptDriftDetector` is responsible for detecting changes in the relationship between inputs and outputs.

**Key Features:**
- Detects changes in model prediction patterns
- Uses statistical methods for drift detection
- Implements online learning for continuous monitoring
- Provides confidence levels for detected drift
- Includes model-agnostic detection methods

**Parameters:**
- `model` (BaseMLModel): The ML model being monitored
- `window_size` (int): Size of the sliding window for drift detection
- `warning_level` (float): Threshold for issuing drift warnings
- `drift_level` (float): Threshold for confirming drift
- `reset_after_drift` (bool): Whether to reset detector after drift

**Methods:**
- `update(X, y)`: Updates the detector with new observations
- `detect_drift()`: Checks if concept drift has occurred
- `get_warning_status()`: Returns if the detector is in warning state
- `get_drift_level()`: Returns the current drift level
- `reset()`: Resets the detector state

### PredictionDriftDetector

`PredictionDriftDetector` is responsible for monitoring changes in the distribution of model predictions over time.

**Key Features:**
- Focuses specifically on model output distribution
- Detects anomalies in prediction patterns
- Supports classification and regression models
- Provides prediction drift metrics and alerts
- Includes confidence intervals for drift detection

**Parameters:**
- `prediction_history` (np.ndarray): Historical prediction data
- `window_size` (int): Size of the window for comparison
- `drift_threshold` (float): Threshold for drift detection
- `model_type` (str): Type of model ('classification', 'regression')
- `alert_on_drift` (bool): Whether to trigger alerts on drift detection

**Methods:**
- `add_predictions(predictions)`: Adds new predictions to the history
- `detect_drift()`: Detects if the prediction distribution has shifted
- `get_drift_metrics()`: Returns detailed drift metrics
- `visualize_prediction_distribution()`: Visualizes prediction distributions
- `reset_history()`: Clears prediction history

### ConceptDriftDetectorEnhanced

`ConceptDriftDetectorEnhanced` is responsible for advanced concept drift detection with multiple algorithms and adaptation strategies.

**Key Features:**
- Combines multiple drift detection algorithms
- Provides drift adaptation strategies
- Implements early warning system
- Supports ensemble-based drift detection
- Includes explainable drift detection

**Parameters:**
- `base_model` (BaseMLModel): Model to monitor for drift
- `drift_detection_methods` (List[str]): Methods to use for drift detection
- `adaptation_strategy` (str): Strategy for adapting to drift
- `sensitivity` (float): Sensitivity level for drift detection
- `explainability` (bool): Whether to provide drift explanations

**Methods:**
- `monitor(X, y)`: Monitors for concept drift in new data
- `get_drift_explanation()`: Explains the nature of detected drift
- `adapt_to_drift(X, y)`: Adapts the model to handle detected drift
- `get_drift_statistics()`: Provides detailed drift statistics
- `visualize_drift_evolution()`: Visualizes drift over time

### FeatureDistributionMonitor

`FeatureDistributionMonitor` is responsible for monitoring the statistical distributions of input features over time.

**Key Features:**
- Tracks distribution metrics for each feature
- Detects gradual and sudden distribution shifts
- Provides temporal analysis of feature stability
- Supports seasonality-aware monitoring
- Generates distribution reports and visualizations

**Parameters:**
- `reference_data` (pd.DataFrame): Baseline data for comparison
- `features_to_monitor` (List[str]): Features to track
- `drift_threshold` (float): Threshold for flagging distribution drift
- `distribution_distance_metric` (str): Metric for measuring distribution distance
- `window_size` (int): Window size for temporal analysis

**Methods:**
- `update(current_data)`: Updates the monitor with new data
- `detect_drift()`: Detects if feature distributions have drifted
- `get_drift_history()`: Returns historical drift measurements
- `get_feature_distribution_metrics()`: Returns distribution metrics for features
- `reset()`: Resets the monitor to initial state

### FeatureCorrelationMonitor

`FeatureCorrelationMonitor` is responsible for tracking changes in correlations between features over time.

**Key Features:**
- Monitors pairwise feature correlations
- Detects changes in correlation patterns
- Identifies emerging and disappearing correlations
- Provides correlation stability metrics
- Generates correlation network visualizations

**Parameters:**
- `reference_data` (pd.DataFrame): Reference data for baseline correlations
- `correlation_threshold` (float): Threshold for significant correlations
- `window_size` (int): Window size for correlation calculations
- `correlation_method` (str): Method for calculating correlations
- `monitor_frequency` (str): How often to calculate correlations

**Methods:**
- `update(new_data)`: Updates correlation calculations with new data
- `detect_correlation_drift()`: Detects if correlation patterns have changed
- `get_correlation_matrix()`: Returns the current correlation matrix
- `get_correlation_history()`: Returns historical correlation measurements
- `identify_changed_correlations()`: Identifies which correlations have changed significantly

### AlertManager

`AlertManager` is responsible for generating, routing, and managing alerts for model monitoring systems.

**Key Features:**
- Centralized alert management for all monitoring components
- Configurable alert severity levels and thresholds
- Multiple notification channels (email, Slack, PagerDuty)
- Alert aggregation and deduplication
- Alert history and resolution tracking

**Parameters:**
- `notification_channels` (Dict[str, Dict]): Configuration for notification channels
- `alert_thresholds` (Dict[str, float]): Thresholds for different alert types
- `throttling_period` (int): Minimum time between similar alerts
- `silent_mode` (bool): Whether to suppress actual notifications (for testing)
- `notification_template_dir` (str, optional): Directory with notification templates

**Methods:**
- `trigger_alert(alert_type, message, details)`: Triggers a new alert
- `resolve_alert(alert_id)`: Marks an alert as resolved
- `get_active_alerts()`: Returns all currently active alerts
- `get_alert_history()`: Returns historical alert records
- `configure_channel(channel, config)`: Configures a notification channel

## Usage Examples

### DataDriftDetector Usage

```python
from app.models.ml.monitoring import DataDriftDetector
import pandas as pd
import numpy as np

# Training data as reference
reference_data = pd.DataFrame({
    'numeric_feature': np.random.normal(0, 1, 1000),
    'categorical_feature': np.random.choice(['A', 'B', 'C'], 1000)
})

# Initialize the detector
detector = DataDriftDetector(
    reference_data=reference_data,
    feature_columns=['numeric_feature', 'categorical_feature'],
    categorical_columns=['categorical_feature'],
    drift_threshold=0.05
)

# New data to check for drift
new_data = pd.DataFrame({
    'numeric_feature': np.random.normal(0.5, 1.2, 200),  # Shifted distribution
    'categorical_feature': np.random.choice(['A', 'B', 'C', 'D'], 200)  # New category
})

# Detect drift
drift_result = detector.detect_drift(new_data)
drifting_features = detector.identify_drifting_features(new_data)

print(f"Drift detected: {drift_result}")
print(f"Drifting features: {drifting_features}")
```

### ConceptDriftDetector Usage

```python
from app.models.ml.monitoring import ConceptDriftDetector
from app.models.ml.prediction import AdScorePredictor
import numpy as np

# Initialize model and drift detector
model = AdScorePredictor()
detector = ConceptDriftDetector(
    model=model,
    window_size=100,
    warning_level=0.05,
    drift_level=0.01
)

# Simulate streaming data
for i in range(1000):
    # Generate data with concept drift after 500 samples
    if i < 500:
        X = np.random.normal(0, 1, (1, 10))
        y = X.sum(axis=1) + np.random.normal(0, 0.1)
    else:
        X = np.random.normal(0, 1, (1, 10))
        y = X.sum(axis=1) * 1.5 + np.random.normal(0, 0.1)  # Concept changed
    
    # Update the detector
    detector.update(X, y)
    
    # Check for drift
    if detector.detect_drift():
        print(f"Concept drift detected at sample {i}")
        
        # Optional: Take action (retrain, alert, etc.)
        if detector.get_drift_level() > 0.5:
            print("Severe drift detected, retraining recommended")
```

## Integration Points

- **Model Registry**: Logs drift metrics for registered models
- **Training Pipeline**: Triggers model retraining when drift is detected
- **Dashboard**: Visualizes monitoring metrics and drift status
- **Alert System**: Sends notifications when anomalies are detected
- **API Gateway**: Provides endpoints for querying monitoring status
- **Data Warehouse**: Stores historical monitoring data for analysis

## Dependencies

- **NumPy/SciPy**: Statistical tests and distributions
- **pandas**: Data handling and feature analysis
- **scikit-learn**: Drift detection algorithms and metrics
- **PyTorch/TensorFlow**: (Optional) Model-specific drift detection
- **Alibi Detect**: Advanced drift detection methods
- **Prometheus/Grafana**: Metrics storage and visualization
- **Apache Kafka**: (Optional) Stream processing for real-time monitoring
