# Production Monitoring Service

**IMPLEMENTATION STATUS: IMPLEMENTED**


## Overview

The Production Monitoring Service is a comprehensive system for monitoring machine learning models in production environments. It provides real-time tracking of model performance, detects data and concept drift, manages alerts, and maintains historical metrics for analysis.

This service is a critical component of the WITHIN ML infrastructure, ensuring that deployed models maintain their performance over time and that stakeholders are promptly notified of any issues.

## Table of Contents

1. [Key Features](#key-features)
2. [Architecture](#architecture)
3. [Installation and Setup](#installation-and-setup)
4. [Configuration](#configuration)
5. [API Reference](#api-reference)
6. [Usage Examples](#usage-examples)
7. [Integration Points](#integration-points)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Performance Considerations](#performance-considerations)

## Key Features

- **Model Performance Tracking**: Monitors key metrics like accuracy, latency, and error rates
- **Drift Detection**: Automatically detects data drift and concept drift
- **Alerting System**: Configurable alerting with multiple channels (log, email, Slack, etc.)
- **Metric Storage**: Historical storage and retrieval of performance metrics
- **Real-time Monitoring**: Continuous monitoring of production predictions
- **Sampling**: Configurable sampling rates to manage monitoring overhead
- **Data Retention**: Automatic cleanup of old monitoring data
- **Health Status**: Overall service and model health reporting

## Architecture

The Production Monitoring Service is implemented as a Python class that integrates with the broader WITHIN ML infrastructure. It consists of the following components:

```
┌─────────────────────────────────────────────────────────────────┐
│                  Production Monitoring Service                   │
│                                                                  │
│  ┌────────────────────┐        ┌─────────────────────────────┐  │
│  │ Model Registration │        │     Metric Collection       │  │
│  └────────────────────┘        └─────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────┐        ┌─────────────────────────────┐  │
│  │  Drift Detection   │        │      Alert Management       │  │
│  └────────────────────┘        └─────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────┐        ┌─────────────────────────────┐  │
│  │   Metric Storage   │        │     Service Management      │  │
│  └────────────────────┘        └─────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Descriptions

- **Model Registration**: Registers models for monitoring with specific configurations
- **Metric Collection**: Records predictions and performance metrics
- **Drift Detection**: Detects changes in data distributions and model behavior
- **Alert Management**: Manages alert generation and routing
- **Metric Storage**: Stores historical metrics and predictions
- **Service Management**: Provides service status and management functions

## Installation and Setup

### Prerequisites

- Python 3.9+
- NumPy, Pandas
- Access to storage location (local filesystem or cloud storage)
- Alert channels configured (if using non-log alerts)

### Basic Setup

```python
from app.services.monitoring.production_monitoring_service import ProductionMonitoringService, ModelMonitoringConfig, AlertLevel

# Initialize with default configuration
monitoring_service = ProductionMonitoringService()

# Or with custom configuration
monitoring_service = ProductionMonitoringService(
    config_path="/path/to/config.json",
    storage_path="/path/to/storage",
    alert_handlers={
        "email": email_alert_handler,
        "slack": slack_alert_handler
    }
)
```

## Configuration

### ModelMonitoringConfig

The `ModelMonitoringConfig` class defines the monitoring configuration for a specific model:

```python
config = ModelMonitoringConfig(
    model_id="ad_score_model_v1",
    performance_metrics=["accuracy", "latency_ms", "error_rate", "auc"],
    drift_detection_interval=30,  # minutes
    performance_threshold={
        "accuracy": 0.92,
        "latency_ms": 150,
        "error_rate": 0.02,
        "auc": 0.85
    },
    alert_channels=["log", "email", "slack"],
    log_predictions=True,
    retention_days=90,
    sampling_rate=0.5  # Sample 50% of predictions for monitoring
)
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_id` | Unique identifier for the model | Required |
| `performance_metrics` | List of metrics to track | ["accuracy", "latency_ms", "error_rate"] |
| `drift_detection_interval` | Interval for drift detection in minutes | 60 |
| `performance_threshold` | Dictionary mapping metrics to threshold values | See below |
| `alert_channels` | List of alert channels | ["log"] |
| `log_predictions` | Whether to log all predictions | True |
| `retention_days` | Number of days to retain monitoring data | 90 |
| `sampling_rate` | Fraction of predictions to sample | 1.0 |

Default performance thresholds:
```python
{
    "accuracy": 0.95,
    "latency_ms": 100,
    "error_rate": 0.01
}
```

### JSON Configuration

You can also provide configuration via JSON file:

```json
{
  "models": [
    {
      "model_id": "ad_score_model_v1",
      "performance_metrics": ["accuracy", "latency_ms", "error_rate", "auc"],
      "drift_detection_interval": 30,
      "performance_threshold": {
        "accuracy": 0.92,
        "latency_ms": 150,
        "error_rate": 0.02,
        "auc": 0.85
      },
      "alert_channels": ["log", "email", "slack"],
      "log_predictions": true,
      "retention_days": 90,
      "sampling_rate": 0.5
    },
    {
      "model_id": "account_health_model_v2",
      "performance_metrics": ["rmse", "mae", "latency_ms"],
      "drift_detection_interval": 60,
      "performance_threshold": {
        "rmse": 0.15,
        "mae": 0.12,
        "latency_ms": 200
      }
    }
  ]
}
```

## API Reference

### Core Methods

#### Model Registration

```python
def register_model(self, config: ModelMonitoringConfig) -> Dict[str, Any]:
    """
    Register a model for monitoring.
    
    Args:
        config: Model monitoring configuration
        
    Returns:
        Dictionary with registration details
    """
```

#### Record Prediction

```python
def record_prediction(
    self,
    model_id: str,
    prediction: Any,
    features: Any,
    actual: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    latency_ms: Optional[float] = None
) -> Dict[str, Any]:
    """
    Record a prediction for monitoring.
    
    Args:
        model_id: ID of the model
        prediction: Model prediction
        features: Input features
        actual: Optional actual value (for calculating accuracy)
        metadata: Optional prediction metadata
        latency_ms: Optional prediction latency in milliseconds
        
    Returns:
        Dictionary indicating success or failure
    """
```

#### Update Model Metric

```python
def update_model_metric(
    self,
    model_id: str,
    metric: str,
    value: float,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Update a metric for a model.
    
    Args:
        model_id: ID of the model
        metric: Name of the metric
        value: Value of the metric
        metadata: Additional metadata
        
    Returns:
        Dictionary with update status
    """
```

#### Get Model Metrics

```python
def get_model_metrics(
    self,
    model_id: str,
    metric: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    aggregation: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get metrics for a model.
    
    Args:
        model_id: ID of the model
        metric: Optional specific metric to retrieve
        start_time: Optional start time for metrics
        end_time: Optional end time for metrics
        aggregation: Optional aggregation method ("mean", "min", "max", "median")
        
    Returns:
        Dictionary with model metrics
    """
```

#### Get Service Status

```python
def get_service_status(self) -> Dict[str, Any]:
    """
    Get the status of the monitoring service.
    
    Returns:
        Dictionary with service status
    """
```

#### Send Alert

```python
def send_alert(
    self,
    severity: AlertLevel, 
    message: str,
    data: Dict[str, Any] = None, 
    alert_type: str = "monitoring"
) -> Dict[str, Any]:
    """
    Send an alert with the provided data.
    
    Args:
        severity: Alert severity level (AlertLevel enum)
        message: Alert message
        data: Alert data
        alert_type: Type of alert
        
    Returns:
        Dictionary with alert data
    """
```

#### Monitor Model Drift

```python
def monitor_model_drift(
    self,
    model_id: str,
    current_data: Optional[Any] = None,
    current_predictions: Optional[np.ndarray] = None,
    current_targets: Optional[np.ndarray] = None,
    comprehensive: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive monitoring for all types of drift in a model.
    
    Args:
        model_id: ID of the model to monitor
        current_data: Current feature data for data drift detection
        current_predictions: Current model predictions for prediction drift detection
        current_targets: Current target values for concept drift detection
        comprehensive: Whether to perform comprehensive analysis
        
    Returns:
        Dictionary with comprehensive drift monitoring results
    """
```

#### Register Alert Handler

```python
def register_alert_handler(self, channel: str, handler: Callable) -> Dict[str, Any]:
    """
    Register an alert handler.
    
    Args:
        channel: Alert channel name
        handler: Handler function
        
    Returns:
        Dictionary indicating success or failure
    """
```

#### Clean Up Old Data

```python
def cleanup_old_data(self) -> Dict[str, Any]:
    """
    Clean up old monitoring data.
    
    Returns:
        Dictionary with cleanup results
    """
```

### AlertLevel Enum

The `AlertLevel` enum defines the severity levels for alerts:

```python
class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
```

## Usage Examples

### Basic Model Registration and Monitoring

```python
from app.services.monitoring.production_monitoring_service import ProductionMonitoringService, ModelMonitoringConfig, AlertLevel
import time
import numpy as np

# Initialize monitoring service
monitoring_service = ProductionMonitoringService(
    storage_path="/data/model_monitoring"
)

# Create configuration for a model
config = ModelMonitoringConfig(
    model_id="ad_score_model_v1",
    performance_metrics=["accuracy", "latency_ms", "prediction_mean"],
    drift_detection_interval=60,  # Check for drift every 60 minutes
    alert_channels=["log"]
)

# Register the model
monitoring_service.register_model(config)

# In your prediction service:
def predict(features):
    start_time = time.time()
    
    # Make prediction
    prediction = model.predict(features)
    
    # Record prediction for monitoring
    latency_ms = (time.time() - start_time) * 1000
    monitoring_service.record_prediction(
        model_id="ad_score_model_v1",
        prediction=prediction,
        features=features,
        latency_ms=latency_ms,
        metadata={
            "user_id": "user_123",
            "platform": "facebook"
        }
    )
    
    return prediction
```

### Custom Alert Handling

```python
def email_alert_handler(alert):
    """Send alert via email."""
    recipient = "ml-team@within.co"
    subject = f"ML Alert [{alert['level']}]: {alert['message']}"
    body = f"""
    Alert Details:
    - Time: {alert['timestamp']}
    - Type: {alert['type']}
    - Level: {alert['level']}
    - Message: {alert['message']}
    
    Additional Data:
    {alert['data']}
    """
    send_email(recipient, subject, body)

# Register the alert handler
monitoring_service.register_alert_handler("email", email_alert_handler)
```

### Monitoring Drift

```python
# Get current production data
current_data = get_recent_production_data()
current_predictions = get_recent_predictions()
current_targets = get_recent_actuals()

# Monitor for drift
drift_result = monitoring_service.monitor_model_drift(
    model_id="ad_score_model_v1",
    current_data=current_data,
    current_predictions=current_predictions,
    current_targets=current_targets
)

# Check drift status
if drift_result["drift_detected"]:
    print(f"Drift detected in: {', '.join(drift_result['drift_types'])}")
    print(f"Severity: {drift_result['severity']}")
    
    # Example: Trigger model retraining if severe drift is detected
    if drift_result["severity"] in ["error", "critical"]:
        trigger_model_retraining("ad_score_model_v1")
```

## Integration Points

### Integration with DriftMonitoringService

The ProductionMonitoringService integrates with the DriftMonitoringService for detailed drift detection:

```python
from app.services.monitoring.drift_monitoring_service import DriftMonitoringService

# Initialize services
production_monitoring = ProductionMonitoringService()
drift_monitoring = DriftMonitoringService()

# Register model in both services
model_config = ModelMonitoringConfig(model_id="ad_score_model_v1")
production_monitoring.register_model(model_config)

drift_config = DriftMonitoringConfig(model_id="ad_score_model_v1")
drift_monitoring.register_model(drift_config)

# In production code:
def process_batch(batch_data, batch_predictions):
    # Record predictions in production monitoring
    for features, prediction in zip(batch_data, batch_predictions):
        production_monitoring.record_prediction(
            model_id="ad_score_model_v1",
            prediction=prediction,
            features=features
        )
    
    # Check for drift at batch level
    drift_result = drift_monitoring.detect_drift(
        model_id="ad_score_model_v1",
        current_data=batch_data,
        current_predictions=batch_predictions
    )
    
    # If drift detected, alert through production monitoring
    if drift_result["drift_detected"]:
        production_monitoring.send_alert(
            severity=AlertLevel.WARNING,
            message=f"Drift detected in model ad_score_model_v1",
            data=drift_result
        )
```

### Integration with Model Training Pipeline

```python
def train_model(model_id, training_data):
    # Train the model
    model = train(training_data)
    
    # Calculate baseline metrics
    baseline_metrics = calculate_metrics(model, training_data)
    
    # Register model with monitoring
    config = ModelMonitoringConfig(
        model_id=model_id,
        performance_metrics=list(baseline_metrics.keys()),
        performance_threshold={
            metric: value * 0.9  # Set threshold at 90% of baseline
            for metric, value in baseline_metrics.items()
        }
    )
    
    monitoring_service.register_model(config)
    
    # Record baseline metrics
    for metric, value in baseline_metrics.items():
        monitoring_service.update_model_metric(
            model_id=model_id,
            metric=metric,
            value=value,
            metadata={"source": "training", "dataset": "baseline"}
        )
    
    return model
```

## Best Practices

### General Recommendations

1. **Configure Appropriate Sampling Rates**: For high-throughput models, use sampling to reduce monitoring overhead.
2. **Set Meaningful Alert Thresholds**: Base thresholds on historical model performance with appropriate margin.
3. **Implement Multiple Alert Channels**: Use at least two channels (e.g., logs and email) for redundancy.
4. **Regular Cleanup**: Schedule regular execution of `cleanup_old_data()` to manage storage.
5. **Monitor the Monitoring**: Track the monitoring service's own performance and resource usage.

### Metric Selection Guidelines

Select metrics that:
- Directly relate to business outcomes
- Capture model-specific performance characteristics
- Can be calculated in near real-time
- Have clear and interpretable thresholds

### Alert Management

1. **Alert Severity Selection**: Choose the appropriate `AlertLevel` based on impact:
   - `INFO`: Informational, no immediate action required
   - `WARNING`: Potential issue, may require attention
   - `ERROR`: Significant issue, requires prompt attention
   - `CRITICAL`: Severe issue, requires immediate attention

2. **Alert Routing**: Route alerts to the appropriate channels based on severity:
   - `INFO`: Logs only
   - `WARNING`: Logs and dashboards
   - `ERROR`: Logs, dashboards, and email
   - `CRITICAL`: Logs, dashboards, email, and real-time notification (e.g., Slack, PagerDuty)

## Troubleshooting

### Common Issues and Solutions

| Issue | Potential Causes | Solutions |
|-------|-----------------|-----------|
| High monitoring overhead | Sampling rate too high | Reduce sampling rate in model configuration |
| Missing metrics | Metric not registered | Ensure metric is in the model's `performance_metrics` list |
| Disk space issues | Data retention period too long | Decrease `retention_days` and run `cleanup_old_data()` |
| False drift alerts | Threshold too sensitive | Adjust drift detection thresholds or sampling window |
| No alerts received | Alert handler not registered | Check alert handler registration and functionality |
| Thread locking issues | Concurrent access problems | Review thread safety in custom implementations |

### Diagnostic Methods

1. **Check service status**:
   ```python
   status = monitoring_service.get_service_status()
   print(status)
   ```

2. **Verify model registration**:
   ```python
   # Should appear in monitored_models list
   status = monitoring_service.get_service_status()
   model_ids = [model["model_id"] for model in status["models"]]
   print(f"Monitored models: {model_ids}")
   ```

3. **Check metric collection**:
   ```python
   metrics = monitoring_service.get_model_metrics(
       model_id="ad_score_model_v1",
       start_time=datetime.now() - timedelta(hours=24)
   )
   print(metrics)
   ```

## Performance Considerations

### Resource Requirements

The Production Monitoring Service's resource requirements depend on several factors:

| Factor | Impact |
|--------|--------|
| Number of models | Linear increase in memory usage |
| Prediction frequency | Linear increase in CPU and I/O |
| Sampling rate | Linear impact on CPU and I/O |
| Drift detection frequency | Periodic CPU spikes |
| Data retention period | Linear impact on disk space |

### Optimization Strategies

1. **Adjust sampling rate** based on model prediction volume
2. **Batch metric updates** for high-frequency models
3. **Schedule drift detection** during off-peak hours
4. **Use efficient serialization** for large feature vectors
5. **Implement storage rotation** for long-term metric retention

### Scaling Guidelines

For high-scale deployments, consider these modifications:

1. **Distributed storage**: Replace file-based storage with a distributed time-series database
2. **Message queue**: Use a message queue for prediction recording
3. **Separate services**: Split drift detection into a separate service
4. **Horizontal scaling**: Deploy multiple monitoring service instances with shared storage
5. **Metrics aggregation**: Implement pre-aggregation for high-volume metrics

---

## Related Documentation

- [Drift Detection Implementation](../drift_detection.md)
- [Monitoring Guide](/docs/maintenance/monitoring_guide.md)
- [Alerting Reference](/docs/maintenance/alerting_reference.md)
- [Model Retraining](/docs/maintenance/model_retraining.md)
- [Troubleshooting Guide](/docs/maintenance/troubleshooting.md) 