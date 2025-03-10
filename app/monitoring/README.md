# Model Monitoring System

This directory contains the monitoring infrastructure for the WITHIN Ad Score & Account Health Predictor system. The monitoring system tracks model performance, detects data drift, and ensures predictions remain accurate and fair over time.

## Directory Structure

```
monitoring/
├── __init__.py                     # Monitoring package initialization
├── performance/                    # Performance monitoring
│   ├── __init__.py                 # Performance package initialization
│   ├── metrics.py                  # Performance metrics calculation
│   ├── thresholds.py               # Performance threshold definitions
│   └── alerting.py                 # Performance alerting system
├── drift/                          # Data drift detection
│   ├── __init__.py                 # Drift package initialization
│   ├── detector.py                 # Data drift detection algorithms
│   ├── statistical.py              # Statistical tests for distributions
│   └── feature_drift.py            # Feature-level drift detection
├── fairness/                       # Fairness monitoring
│   ├── __init__.py                 # Fairness package initialization
│   ├── metrics.py                  # Fairness metric calculations
│   ├── assessor.py                 # Fairness assessment implementation
│   └── mitigator.py                # Bias mitigation strategies
├── logging/                        # Prediction logging
│   ├── __init__.py                 # Logging package initialization
│   ├── prediction_logger.py        # Prediction event logging
│   ├── structured_logger.py        # Structured logging utilities
│   └── audit.py                    # Audit trail implementation
├── visualization/                  # Monitoring visualizations
│   ├── __init__.py                 # Visualization package initialization
│   ├── dashboards.py               # Monitoring dashboard generation
│   ├── drift_plots.py              # Data drift visualization
│   └── performance_plots.py        # Performance visualization
├── storage/                        # Metrics storage
│   ├── __init__.py                 # Storage package initialization
│   ├── metrics_store.py            # Performance metrics storage
│   ├── baseline_store.py           # Baseline distribution storage
│   └── prediction_store.py         # Prediction history storage
└── monitors/                       # Monitor implementations
    ├── __init__.py                 # Monitors package initialization
    ├── ad_score_monitor.py         # Ad score model monitor
    ├── account_health_monitor.py   # Account health model monitor
    ├── system_monitor.py           # System performance monitor
    └── retraining_scheduler.py     # Model retraining scheduler
```

## Core Components

### Model Monitors

The monitoring system provides specialized monitors for each model type:

#### Ad Score Model Monitor

```python
from app.monitoring.monitors import AdScoreModelMonitor

# Initialize monitor
monitor = AdScoreModelMonitor(model_id="ad_score_model_v1")

# Check model health
health_report = monitor.check_model_health(
    recent_data=recent_features,
    predictions=recent_predictions,
    actuals=recent_actuals
)

# Extract findings
if health_report["status"] != "healthy":
    findings = health_report["findings"]
    for finding in findings:
        print(f"Issue: {finding['type']}, Severity: {finding['severity']}")
        print(f"Description: {finding['description']}")
```

Example health report:

```json
{
    "model_id": "ad_score_model_v1",
    "timestamp": "2023-04-15T10:30:45Z",
    "status": "warning",
    "findings": [
        {
            "type": "data_drift",
            "severity": "medium",
            "description": "Data drift detected in 3 features: ctr_history, ad_length, sentiment_score..."
        }
    ],
    "metrics": {
        "drift": {
            "drift_detected": true,
            "drifted_features": ["ctr_history", "ad_length", "sentiment_score"],
            "feature_metrics": {
                "ctr_history": {
                    "kl_divergence": 0.25,
                    "psi": 0.28,
                    "drift_detected": true
                },
                "ad_length": {
                    "kl_divergence": 0.22,
                    "psi": 0.26,
                    "drift_detected": true
                },
                "sentiment_score": {
                    "kl_divergence": 0.21,
                    "psi": 0.27,
                    "drift_detected": true
                }
            }
        },
        "performance": {
            "overall": {
                "rmse": 9.2,
                "mae": 7.1,
                "r2": 0.82
            },
            "segments": {
                "high_value": {
                    "rmse": 10.5,
                    "count": 245
                },
                "medium_value": {
                    "rmse": 8.3,
                    "count": 512
                },
                "low_value": {
                    "rmse": 8.9,
                    "count": 243
                }
            }
        }
    }
}
```

### Data Drift Detection

The drift detection system monitors input feature distributions for changes that could affect model performance:

```python
from app.monitoring.drift import DataDriftDetector

# Initialize detector
detector = DataDriftDetector(
    reference_data=baseline_data,
    drift_thresholds={
        "kl_divergence": 0.2,
        "psi": 0.25,
        "js_divergence": 0.15
    }
)

# Detect drift in current data
drift_results = detector.detect_drift(current_data)

# Take action based on results
if drift_results["drift_detected"]:
    drifted_features = drift_results["drifted_features"]
    print(f"Data drift detected in {len(drifted_features)} features")
    # Trigger retraining or alerting
```

### Fairness Monitoring

The fairness monitoring system ensures predictions are fair across protected groups:

```python
from app.monitoring.fairness import FairnessAssessor

# Initialize fairness assessor
assessor = FairnessAssessor(
    protected_attributes=["age_group", "gender", "location_type", "language"]
)

# Assess fairness
fairness_report = assessor.assess_fairness(
    predictions=model_predictions,
    actual_values=actual_values,
    metadata=user_metadata
)

# Implement mitigation strategies if needed
if not fairness_report["overall_fairness"]:
    for recommendation in fairness_report["recommendations"]:
        print(f"Issue with {recommendation['attribute']}: {recommendation['description']}")
        # Implement the recommended mitigation strategy
```

### Performance Tracking

The performance tracking system monitors model performance metrics over time:

```python
from app.monitoring.performance import PerformanceTracker

# Initialize tracker
tracker = PerformanceTracker(model_id="ad_score_model_v1")

# Log new performance metrics
tracker.log_metrics(
    metrics={
        "rmse": 8.7,
        "mae": 6.8,
        "r2": 0.85
    },
    segment="all",
    timestamp=datetime.now()
)

# Get performance history
performance_history = tracker.get_metrics_history(
    start_date="2023-01-01",
    end_date="2023-04-15",
    metrics=["rmse", "r2"]
)

# Check for performance degradation
degradation = tracker.check_degradation(
    current_metrics={
        "rmse": 9.2,
        "mae": 7.1,
        "r2": 0.82
    },
    threshold_percent=10
)
```

### Retraining Scheduling

The retraining scheduler determines when models should be retrained:

```python
from app.monitoring.monitors import RetrainingScheduler

# Initialize scheduler
scheduler = RetrainingScheduler()

# Determine if retraining is needed
retraining_decision = scheduler.schedule_retraining(
    model_id="ad_score_model_v1",
    drift_results=drift_results,
    performance_results=performance_results
)

if retraining_decision["should_retrain"]:
    print(f"Model retraining recommended with {retraining_decision['urgency']} urgency")
    print(f"Reasons: {', '.join(retraining_decision['reasons'])}")
    # Trigger retraining pipeline
```

## Monitoring Dashboard

The monitoring system includes dashboards for visualizing model performance:

```python
from app.monitoring.visualization import MonitoringDashboardGenerator

# Generate dashboard
dashboard = MonitoringDashboardGenerator.create_dashboard(
    model_id="ad_score_model_v1",
    start_date="2023-01-01",
    end_date="2023-04-15"
)

# Save or display dashboard
dashboard.save("ad_score_model_dashboard.html")
```

The dashboard includes:

1. **Performance Metrics**: Trends of key metrics over time
2. **Data Drift**: Distribution comparisons between baseline and current data
3. **Prediction Distribution**: Changes in prediction distributions
4. **Fairness Metrics**: Fairness metric trends over time
5. **Error Analysis**: Error distribution and problematic segments
6. **Resource Utilization**: CPU, memory, and latency metrics

## Alerting System

The monitoring system can send alerts when issues are detected:

```python
from app.monitoring.performance import AlertManager

# Initialize alert manager
alert_manager = AlertManager()

# Send alert
alert_manager.send_alert(
    level="warning",
    model_id="ad_score_model_v1",
    findings=[
        {
            "type": "data_drift",
            "severity": "medium",
            "description": "Data drift detected in 3 features"
        }
    ]
)
```

Alert channels include:

- Email notifications
- Slack messages
- PagerDuty incidents
- Dashboard notifications
- Logging to monitoring systems

## Metrics Storage

Performance metrics and baselines are stored for historical comparison:

```python
from app.monitoring.storage import MetricsStore

# Initialize metrics store
metrics_store = MetricsStore()

# Store metrics
metrics_store.store_metrics(
    model_id="ad_score_model_v1",
    metrics={
        "rmse": 8.7,
        "mae": 6.8,
        "r2": 0.85
    },
    timestamp=datetime.now()
)

# Retrieve historical metrics
historical_metrics = metrics_store.get_metrics(
    model_id="ad_score_model_v1",
    start_date="2023-01-01",
    end_date="2023-04-15",
    metrics=["rmse", "r2"]
)

# Store baseline distributions
metrics_store.store_baseline(
    model_id="ad_score_model_v1",
    feature_distributions=feature_distributions,
    prediction_distribution=prediction_distribution,
    performance_metrics=performance_metrics
)
```

## Development Guidelines

When enhancing or adding monitoring components:

1. **Automated Performance Tracking**: Implement metrics calculation that runs automatically
2. **Granular Monitoring**: Track metrics with appropriate time granularity
3. **Alerting Setup**: Configure alerting for significant performance degradation
4. **Baseline Documentation**: Document baseline performance for all models
5. **Distribution Monitoring**: Implement statistical tests for distribution shifts
6. **Prediction Tracking**: Monitor prediction distribution changes over time
7. **Feature Correlation**: Maintain feature correlation stability monitoring
8. **Regular Health Checks**: Schedule model health assessments
9. **Resource Tracking**: Monitor inference latency and resource utilization
10. **Edge Case Monitoring**: Specifically monitor performance on edge cases
11. **Clear Retraining Triggers**: Define explicit triggers for model retraining 