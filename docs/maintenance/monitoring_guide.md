# Monitoring Guide

**IMPLEMENTATION STATUS: IMPLEMENTED**


This guide provides comprehensive instructions for monitoring WITHIN's ML models and systems in production. Effective monitoring is essential for maintaining model accuracy, detecting drift, ensuring system health, and providing timely alerts for potential issues.

## Table of Contents

- [Overview](#overview)
- [Monitoring Architecture](#monitoring-architecture)
- [Model Performance Monitoring](#model-performance-monitoring)
- [System Performance Monitoring](#system-performance-monitoring)
- [Data Quality Monitoring](#data-quality-monitoring)
- [Setting Up Alerts](#setting-up-alerts)
- [Dashboards](#dashboards)
- [Incident Response](#incident-response)
- [Scheduled Reporting](#scheduled-reporting)

## Overview

The WITHIN monitoring system consists of several integrated components that track:

1. **Model Performance**: Accuracy, predictions distribution, feature importance stability
2. **System Performance**: Latency, throughput, resource utilization, errors
3. **Data Quality**: Drift detection, schema validation, missing values
4. **User Activity**: Usage patterns, endpoint popularity, client errors

Monitoring data is collected in real-time, aggregated, and visualized in customizable dashboards. Automated alerts notify engineering teams when metrics exceed defined thresholds.

For detailed implementation documentation of the monitoring components, see:
- [Production Monitoring Service](/docs/implementation/ml/monitoring/production_monitoring_service.md)
- [Drift Detection Implementation](/docs/implementation/ml/drift_detection.md)

## Monitoring Architecture

![Monitoring Architecture](/docs/images/monitoring_architecture.png)

The monitoring architecture consists of:

- **Collectors**: Components that gather raw metrics from various sources
- **Storage**: Time-series databases for metrics and logs
- **Processors**: Components that analyze, aggregate, and detect anomalies
- **Visualization**: Dashboards for exploring monitoring data
- **Alerting**: Rules-based notification system

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Metrics Collection | Prometheus | Collect system and application metrics |
| Log Collection | Fluentd | Aggregate logs from all services |
| Metrics Storage | InfluxDB | Store time-series metrics |
| Log Storage | Elasticsearch | Store and index logs |
| Visualization | Grafana | Create monitoring dashboards |
| Alerting | AlertManager | Manage and route alerts |
| ML Monitoring | Custom Service | Monitor model-specific metrics |

## Model Performance Monitoring

### Key Metrics to Track

#### Ad Score Predictor
- Prediction distribution (daily)
- Accuracy against known outcomes
- Feature importance stability
- Confidence score distribution
- Latency (p50, p95, p99)

#### Account Health Predictor
- Health score distribution by platform
- Recommendation relevance
- Anomaly detection precision/recall
- Forecast accuracy vs. actual
- Processing time trends

#### Ad Sentiment Analyzer
- Sentiment distribution trends
- Aspect scores distribution
- Confidence score distribution
- User feedback on sentiment accuracy
- Processing time for various text lengths

### Setting Up Model Monitoring

```python
from within.monitoring import ModelMonitor

# Initialize model monitor
ad_score_monitor = ModelMonitor(
    model_name="ad_score_predictor",
    version="2.1.0",
    metrics=[
        "prediction_distribution",
        "confidence_distribution",
        "feature_importance",
        "latency"
    ]
)

# Register callbacks for live monitoring
@ad_score_monitor.on_prediction
def track_prediction(prediction, input_data):
    """Log each prediction for monitoring."""
    ad_score_monitor.log_prediction(
        prediction_value=prediction["score"],
        confidence=prediction["confidence"],
        feature_values=input_data,
        metadata={
            "platform": input_data.get("platform"),
            "audience": input_data.get("target_audience")
        }
    )

# Configure drift detection
ad_score_monitor.configure_drift_detection(
    reference_data="2023-09-01",  # Reference distribution date
    drift_metrics=["wasserstein", "ks_test"],
    feature_importance_threshold=0.1,  # Alert if feature importance changes by 10%
    prediction_drift_threshold=0.05  # Alert if distribution shifts by 5%
)

# Enable the monitor
ad_score_monitor.enable()
```

### Accessing Model Metrics API

```python
# Get model performance metrics
import requests

api_url = "https://api.within.co/api/v1/monitoring/models/ad_score_predictor"
headers = generate_auth_headers(access_key, secret_key, "GET", api_url)

response = requests.get(
    api_url,
    headers=headers,
    params={
        "start_date": "2023-10-01",
        "end_date": "2023-10-15",
        "metrics": "accuracy,latency,drift"
    }
)

metrics = response.json()
print(f"Average accuracy: {metrics['data']['accuracy']['average']}")
print(f"Drift detected: {metrics['data']['drift']['detected']}")
```

## System Performance Monitoring

### Key System Metrics

| Metric Category | Specific Metrics | Warning Threshold | Critical Threshold |
|-----------------|------------------|-------------------|-------------------|
| **API Performance** | Request rate | 80% of capacity | 90% of capacity |
| | Latency (p95) | 300ms | 500ms |
| | Error rate | 1% | 5% |
| **Resource Utilization** | CPU usage | 70% | 85% |
| | Memory usage | 75% | 90% |
| | Disk usage | 70% | 85% |
| **Database** | Query latency | 100ms | 300ms |
| | Connection pool usage | 70% | 85% |
| | Cache hit ratio | Below 80% | Below 60% |
| **Batch Processing** | Job success rate | Below 98% | Below 95% |
| | Processing time | 2x baseline | 3x baseline |
| | Queue depth | 1000 items | 5000 items |

### Monitoring Endpoints

The system provides the following monitoring endpoints:

- `/health`: Basic health check (returns 200 if system is operational)
- `/metrics`: Prometheus-formatted metrics
- `/status`: Detailed component status
- `/monitoring/performance`: System performance dashboard

### System Dashboard Setup

System dashboards are available through Grafana at:
- Production: https://monitoring.within.co/grafana
- Staging: https://monitoring-staging.within.co/grafana

Default dashboards include:
- API Performance
- Resource Utilization
- Model Performance
- Batch Processing
- Data Pipeline

## Data Quality Monitoring

### Data Drift Detection

Data drift can significantly impact model performance. The system monitors:

1. **Feature Drift**: Changes in the statistical properties of input features
2. **Prediction Drift**: Changes in the distribution of model predictions
3. **Feature Relationship Drift**: Changes in correlations between features

```python
from within.monitoring import DataDriftMonitor

# Initialize drift monitor
drift_monitor = DataDriftMonitor(
    reference_dataset="2023-09-baseline",
    features_to_monitor=[
        "ad_content.length", 
        "historical_metrics.ctr",
        "audience_size"
    ],
    statistical_tests=[
        "kolmogorov_smirnov", 
        "wasserstein_distance"
    ]
)

# Check for drift in current data
drift_results = drift_monitor.check_drift(current_data)

if drift_results.drift_detected:
    print(f"Data drift detected! Metrics: {drift_results.metrics}")
    print(f"Features with drift: {drift_results.drifted_features}")
    
    # Optionally trigger retraining
    if drift_results.severity > 0.5:
        trigger_model_retraining()
```

### Data Validation

The system continuously validates incoming data against expected schemas and constraints:

```python
from within.validation import DataValidator

# Define validation rules
validator = DataValidator()
validator.add_rule("ad_content.headline", "max_length", 150)
validator.add_rule("ad_content.description", "max_length", 500)
validator.add_rule("historical_metrics.ctr", "range", min=0.0, max=1.0)
validator.add_rule("platform", "enum", ["facebook", "google", "tiktok", "linkedin"])

# Validate incoming data
validation_result = validator.validate(incoming_data)

if not validation_result.is_valid:
    print(f"Validation errors: {validation_result.errors}")
    # Handle invalid data
```

### Data Quality Dashboard

The Data Quality Dashboard provides visualizations for:
- Feature distributions over time
- Data completeness metrics
- Validation failure rates
- Drift detection alerts
- Schema evolution

## Setting Up Alerts

### Alert Channels

The system supports multiple alert channels:

- Email notifications
- Slack alerts
- PagerDuty integration
- SMS alerts (for critical issues)
- Dashboard notifications

### Configuring Alerts

Alerts can be configured via the Alerts Management UI or programmatically:

```python
from within.monitoring import AlertManager

# Initialize alert manager
alert_manager = AlertManager()

# Configure an alert for model accuracy drop
alert_manager.create_alert(
    name="ad_score_accuracy_drop",
    description="Alert when Ad Score Predictor accuracy drops below threshold",
    metric="models.ad_score_predictor.accuracy",
    condition="< 0.80",
    window="1h",
    severity="warning",
    channels=["slack-ml-team", "email-ml-engineers"],
    notification_template="Model accuracy drop: {metric} is now {value}, below {threshold}"
)

# Configure system alert
alert_manager.create_alert(
    name="api_high_latency",
    description="Alert when API latency exceeds threshold",
    metric="api.latency.p95",
    condition="> 300",
    window="5m",
    severity="critical",
    channels=["slack-ops", "pagerduty-oncall"],
    notification_template="API latency alert: p95 latency is {value}ms (threshold: {threshold}ms)"
)
```

### Alert Severity Levels

| Severity | Description | Response Time | Notification Channels |
|----------|-------------|---------------|------------------------|
| **Info** | Informational event | No action required | Dashboard, Email digest |
| **Warning** | Potential issue requiring attention | < 24 hours | Slack, Email |
| **High** | Significant issue requiring prompt attention | < 4 hours | Slack, Email, SMS |
| **Critical** | Severe issue requiring immediate action | < 30 minutes | Slack, PagerDuty, SMS, Phone call |

## Dashboards

### Main Monitoring Dashboard

The main monitoring dashboard is accessible at https://monitoring.within.co and provides an overview of all system components.

![Main Dashboard](/docs/images/monitoring_dashboard.png)

### Model-Specific Dashboards

Each ML model has a dedicated dashboard:

- [Ad Score Predictor Dashboard](https://monitoring.within.co/models/ad-score)
- [Account Health Predictor Dashboard](https://monitoring.within.co/models/account-health)
- [Ad Sentiment Analyzer Dashboard](https://monitoring.within.co/models/sentiment)

### Custom Dashboard Creation

Custom dashboards can be created using Grafana:

1. Log into the Grafana instance at https://monitoring.within.co/grafana
2. Navigate to "Create" > "Dashboard"
3. Add panels using the available metrics
4. Save and share the dashboard

Example custom dashboard configuration:

```json
{
  "dashboard": {
    "id": null,
    "title": "Ad Score Performance Tracking",
    "tags": ["ad-score", "ml-performance"],
    "panels": [
      {
        "title": "Ad Score Distribution",
        "type": "histogram",
        "datasource": "InfluxDB",
        "targets": [
          {
            "query": "SELECT score FROM ad_score_predictions WHERE time > now() - 7d",
            "refId": "A"
          }
        ]
      },
      {
        "title": "Prediction Accuracy",
        "type": "graph",
        "datasource": "InfluxDB",
        "targets": [
          {
            "query": "SELECT mean(accuracy) FROM model_metrics WHERE model='ad_score_predictor' AND time > now() - 30d GROUP BY time(1d)",
            "refId": "A"
          }
        ]
      }
    ],
    "time": {
      "from": "now-7d",
      "to": "now"
    }
  }
}
```

## Incident Response

### Incident Classification

| Level | Description | Example | Response |
|-------|-------------|---------|----------|
| P1 | Critical service disruption | API down, predictions failing | Immediate response, all hands |
| P2 | Major functionality affected | High latency, partial outage | Urgent response, dedicated team |
| P3 | Minor issue with workaround | Accuracy degradation, non-critical errors | Same-day response |
| P4 | Cosmetic or low-impact issue | UI issues, minor bugs | Scheduled fix |

### Incident Response Workflow

1. **Detection**: Alert triggered or issue reported
2. **Triage**: Assess severity and impact
3. **Notification**: Inform stakeholders based on severity
4. **Investigation**: Identify root cause
5. **Mitigation**: Implement temporary fix if needed
6. **Resolution**: Deploy permanent solution
7. **Post-mortem**: Document incident and preventive measures

### Incident Documentation

All incidents should be documented with:
- Date and time of occurrence
- Detection method (alert, user report, etc.)
- Impact assessment
- Timeline of events
- Root cause analysis
- Resolution steps
- Preventive measures

## Scheduled Reporting

### Standard Reports

The following reports are automatically generated:

| Report | Frequency | Recipients | Contents |
|--------|-----------|------------|----------|
| System Health | Daily | Engineering Team | System performance, errors, resource usage |
| Model Performance | Weekly | ML Team, Product Managers | Accuracy metrics, drift analysis, usage statistics |
| Data Quality | Weekly | Data Team, ML Team | Data validation results, schema changes, drift detection |
| Incident Summary | Monthly | Leadership, Engineering | Incident count, resolution times, impact assessment |

### Configuring Custom Reports

Custom reports can be configured using the reporting API:

```python
from within.reporting import ReportManager

# Initialize report manager
report_manager = ReportManager()

# Create custom report
report_manager.create_report(
    name="ad_score_platform_comparison",
    description="Weekly comparison of Ad Score performance across platforms",
    schedule="0 9 * * MON",  # 9 AM every Monday
    format="pdf",
    recipients=["marketing-team@company.com", "ml-team@company.com"],
    content={
        "sections": [
            {
                "title": "Ad Score Accuracy by Platform",
                "type": "chart",
                "metric": "models.ad_score_predictor.accuracy",
                "dimensions": ["platform"],
                "period": "7d"
            },
            {
                "title": "Usage Statistics",
                "type": "table",
                "metrics": ["request_count", "unique_users"],
                "dimensions": ["platform", "endpoint"],
                "period": "7d"
            }
        ]
    }
)
```

## Additional Resources

- [Model Retraining Guide](./model_retraining.md): Instructions for retraining models
- [Troubleshooting Guide](./troubleshooting.md): Resolving common issues
- [Monitoring API Reference](../api/monitoring_api.md): API documentation for monitoring
- [Alerting Reference](../maintenance/alerting_reference.md): Complete alert configuration options
- [Dashboard Templates](../resources/dashboard_templates/): Pre-built Grafana dashboard templates 