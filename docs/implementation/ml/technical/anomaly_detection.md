# Anomaly Detection Methodology

**IMPLEMENTATION STATUS: IMPLEMENTED**


This document describes the methodology used for anomaly detection in the WITHIN platform, particularly in the context of the Account Health Predictor and ML monitoring systems.

## Table of Contents

1. [Introduction](#introduction)
2. [Anomaly Detection Approaches](#anomaly-detection-approaches)
3. [Statistical Methods](#statistical-methods)
4. [Implementation Details](#implementation-details)
5. [Monitoring Integration](#monitoring-integration)
6. [Best Practices](#best-practices)
7. [Future Improvements](#future-improvements)

## Introduction

Anomaly detection is a critical component of WITHIN's ML monitoring and account health prediction systems. It enables:

- Early detection of irregular patterns in model performance metrics
- Identification of unusual account behavior that may indicate problems
- Automated alerting when metrics deviate from expected ranges
- Performance degradation detection before it impacts business outcomes

In the context of the Account Health Predictor, anomaly detection serves as both:

1. A feature engineering component that generates input signals for the prediction model
2. A monitoring mechanism that ensures the reliability of the prediction system itself

This document focuses on the technical methodologies used for anomaly detection, their implementation details, and integration with other components.

## Anomaly Detection Approaches

The WITHIN platform employs multiple approaches to anomaly detection, each suited for different use cases:

### Statistical Anomaly Detection

This approach relies on statistical properties of the data to identify outliers. The primary methods used include:

1. **Z-Score Analysis**: Identifying data points that deviate significantly from the mean
2. **Moving Window Analysis**: Using recent history as a baseline for detecting changes
3. **Distribution-Based Methods**: Analyzing full distributions rather than summary statistics

### Machine Learning-Based Detection

For more complex patterns, supervised and unsupervised ML methods are employed:

1. **Isolation Forests**: Efficient for high-dimensional data
2. **One-Class SVM**: Effective for well-defined normal distributions
3. **Autoencoders**: Useful for learning complex normal patterns

### Business Rule-Based Detection

Domain-specific rules derived from business knowledge:

1. **Threshold-Based Rules**: Simple but effective for well-understood metrics
2. **Multi-Condition Rules**: Combinations of conditions that indicate anomalies
3. **Seasonal Adjustment**: Accounting for expected cyclical patterns

## Statistical Methods

### Z-Score Analysis

The primary statistical method used in the monitoring system is Z-score analysis, which measures how many standard deviations a data point is from the mean of a baseline distribution.

#### Mathematical Definition

For a data point \(x\), the Z-score is calculated as:

\[ Z = \frac{x - \mu}{\sigma} \]

Where:
- \(\mu\) is the mean of the baseline distribution
- \(\sigma\) is the standard deviation of the baseline distribution

#### Implementation

The implementation uses a sliding window approach to calculate the baseline statistics:

```python
def detect_anomalies(monitoring_data: pd.DataFrame, metrics: List[str] = None, 
                    sensitivity: float = 2.0, window_size: int = 24) -> Dict[str, Any]:
    """
    Detect anomalies in monitoring metrics.
    
    Args:
        monitoring_data: DataFrame containing monitoring data
        metrics: List of metrics to check for anomalies
        sensitivity: Z-score threshold for anomaly detection
        window_size: Window size for baseline calculation
        
    Returns:
        Dictionary with detected anomalies
    """
    if metrics is None:
        metrics = monitoring_data.columns.tolist()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "anomalies": {},
        "total_metrics_checked": len(metrics),
        "total_anomalies_found": 0,
        "anomalies_detected": False,
        "anomalous_points": []
    }
    
    for metric in metrics:
        if metric not in monitoring_data.columns:
            continue
        
        # Get values for the metric
        values = monitoring_data[metric].values
        
        if len(values) < window_size + 1:
            # Not enough data
            continue
        
        # Calculate baseline statistics using historical window
        baseline = values[-(window_size+1):-1]
        baseline_mean = np.mean(baseline)
        baseline_std = np.std(baseline)
        
        if baseline_std == 0:
            # Skip metrics with no variation
            continue
        
        # Calculate z-score for most recent point
        current = values[-1]
        z_score = abs(current - baseline_mean) / baseline_std
        
        # Check if anomaly
        is_anomaly = z_score > sensitivity
        
        if is_anomaly:
            results["anomalies"][metric] = {
                "value": float(current),
                "baseline_mean": float(baseline_mean),
                "baseline_std": float(baseline_std),
                "z_score": float(z_score),
                "threshold": float(sensitivity)
            }
            results["total_anomalies_found"] += 1
            results["anomalies_detected"] = True
            
            # Add to anomalous points
            results["anomalous_points"].append({
                "timestamp": monitoring_data.loc[len(monitoring_data)-1, "timestamp"] if "timestamp" in monitoring_data.columns else datetime.now().isoformat(),
                "metric": metric,
                "value": float(current),
                "expected_range": [float(baseline_mean - sensitivity * baseline_std), float(baseline_mean + sensitivity * baseline_std)]
            })
    
    return results
```

#### Key Parameters

- **sensitivity**: The Z-score threshold for flagging an anomaly (default: 2.0)
- **window_size**: The number of historical data points to use as baseline (default: 24)

The sensitivity parameter allows for tuning the detector's threshold. A value of 2.0 means data points more than 2 standard deviations from the mean are flagged as anomalies, which corresponds to roughly 5% of observations in a normal distribution.

### Moving Average Anomaly Detection

For metrics with trends, a moving average approach is often more appropriate:

```python
def detect_trend_anomalies(time_series: np.ndarray, window_size: int = 5, 
                          sensitivity: float = 3.0) -> List[int]:
    """
    Detect anomalies in a time series using moving average.
    
    Args:
        time_series: Array of time series values
        window_size: Window for calculating moving average
        sensitivity: Threshold for anomaly detection
        
    Returns:
        List of indices with anomalies
    """
    if len(time_series) < window_size * 2:
        return []
        
    # Calculate moving average
    moving_avg = np.convolve(time_series, np.ones(window_size)/window_size, mode='valid')
    
    # Pad the moving average array to match original length
    pad_size = len(time_series) - len(moving_avg)
    moving_avg = np.pad(moving_avg, (pad_size, 0), 'edge')
    
    # Calculate residuals (difference from moving average)
    residuals = time_series - moving_avg
    
    # Calculate standard deviation of residuals
    residual_std = np.std(residuals)
    
    if residual_std == 0:
        return []
    
    # Identify anomalies
    anomalies = []
    for i, residual in enumerate(residuals):
        if abs(residual) > sensitivity * residual_std:
            anomalies.append(i)
    
    return anomalies
```

This approach is especially useful for metrics that exhibit trends or seasonal patterns.

## Implementation Details

### Anomaly Detection Engine

The core anomaly detection engine in WITHIN is implemented as a modular system that supports multiple detection methods and configurations.

#### Core Classes

1. **AnomalyDetector**: Base interface for all anomaly detectors
2. **StatisticalAnomalyDetector**: Implementation using statistical methods
3. **MLAnomalyDetector**: Implementation using machine learning techniques
4. **AnomalyDetectionEngine**: Orchestrator that manages multiple detectors

#### Class Diagram

```
┌───────────────────────┐
│  AnomalyDetector      │
│  (Interface)          │
├───────────────────────┤
│ + detect(data)        │
│ + train(data)         │
│ + get_anomaly_score() │
└─────────┬─────────────┘
          │
          ├─────────────────┬────────────────────┐
          │                 │                    │
┌─────────▼─────────┐ ┌─────▼──────────┐ ┌──────▼───────────┐
│ Statistical       │ │ ML-based       │ │ Rule-based       │
│ AnomalyDetector   │ │ AnomalyDetector│ │ AnomalyDetector  │
└─────────┬─────────┘ └────────────────┘ └──────────────────┘
          │
          ├─────────────────┬────────────────────┐
          │                 │                    │
┌─────────▼─────────┐ ┌─────▼──────────┐ ┌──────▼───────────┐
│ ZScoreDetector    │ │ MADDetector    │ │ QuantileDetector │
└───────────────────┘ └────────────────┘ └──────────────────┘
```

#### Configuration System

The anomaly detection system is highly configurable through a YAML-based configuration:

```yaml
anomaly_detection:
  default_method: z_score
  sensitivity: 2.5
  methods:
    z_score:
      window_size: 24
      min_data_points: 10
    mad:  # Median Absolute Deviation
      window_size: 48
      threshold_multiplier: 3.0
    isolation_forest:
      contamination: 0.05
      n_estimators: 100
  metrics:
    error_rate:
      method: z_score
      sensitivity: 3.0
      window_size: 12
    latency_p95:
      method: mad
      window_size: 24
```

### Integration with Account Health Predictor

The Account Health Predictor leverages anomaly detection in two ways:

1. **Feature Generation**: Anomaly scores are used as input features
2. **Triggering Mechanism**: Anomalies can trigger health assessment

#### Feature Generation

```python
def generate_anomaly_features(account_metrics: pd.DataFrame, 
                             config: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate anomaly-based features for Account Health Prediction.
    
    Args:
        account_metrics: DataFrame with historical account metrics
        config: Configuration for anomaly detection
        
    Returns:
        DataFrame with anomaly features
    """
    features = pd.DataFrame(index=account_metrics.index)
    
    # Configure detectors
    detectors = {}
    for metric, settings in config.get("metrics", {}).items():
        if metric in account_metrics.columns:
            method = settings.get("method", config.get("default_method", "z_score"))
            params = settings.copy()
            params.pop("method", None)
            
            # Create detector based on method
            if method == "z_score":
                detectors[metric] = ZScoreDetector(**params)
            elif method == "mad":
                detectors[metric] = MADDetector(**params)
            # Add other methods as needed
    
    # Calculate anomaly scores for each metric
    for metric, detector in detectors.items():
        # Fit detector on historical data
        detector.fit(account_metrics[metric].values)
        
        # Get anomaly scores
        scores = detector.score(account_metrics[metric].values)
        
        # Add as feature
        features[f"{metric}_anomaly_score"] = scores
        
        # Add binary anomaly flags
        features[f"{metric}_is_anomaly"] = scores > detector.threshold
        
        # Add distance from threshold
        features[f"{metric}_threshold_distance"] = scores / detector.threshold - 1.0
    
    # Add aggregate features
    if len(detectors) > 0:
        features["total_anomalies"] = features.filter(like="_is_anomaly").sum(axis=1)
        features["max_anomaly_score"] = features.filter(like="_anomaly_score").max(axis=1)
        features["mean_anomaly_score"] = features.filter(like="_anomaly_score").mean(axis=1)
    
    return features
```

## Monitoring Integration

The anomaly detection system is integrated with the ProductionMonitoringService for real-time monitoring of model performance and alerting.

### Monitoring Workflow

1. Metrics are collected through the monitoring service
2. Anomaly detection is performed on these metrics
3. Alerts are generated when anomalies are detected
4. Response actions are triggered based on the severity

```python
# Example integration with monitoring service
def monitor_metrics(metrics_data: pd.DataFrame, 
                   model_id: str, 
                   anomaly_config: Dict[str, Any]) -> None:
    """
    Monitor metrics for anomalies and trigger alerts.
    
    Args:
        metrics_data: DataFrame with monitoring metrics
        model_id: ID of the model being monitored
        anomaly_config: Anomaly detection configuration
    """
    # Get monitoring service instance
    monitoring_service = ProductionMonitoringService()
    
    # Detect anomalies
    anomaly_result = detect_anomalies(
        monitoring_data=metrics_data,
        metrics=anomaly_config.get("metrics"),
        sensitivity=anomaly_config.get("sensitivity", 2.0),
        window_size=anomaly_config.get("window_size", 24)
    )
    
    # Log anomaly check
    monitoring_service._log_monitoring_result(
        model_id=model_id,
        result_type="anomaly_detection",
        result=anomaly_result
    )
    
    # Send alerts if anomalies detected
    if anomaly_result["anomalies_detected"]:
        # Determine severity based on number of anomalies
        if anomaly_result["total_anomalies_found"] > 3:
            severity = AlertLevel.CRITICAL
        elif anomaly_result["total_anomalies_found"] > 1:
            severity = AlertLevel.ERROR
        else:
            severity = AlertLevel.WARNING
            
        # Send alert
        monitoring_service.send_alert(
            severity=severity,
            message=f"Anomalies detected in {model_id} metrics",
            data=anomaly_result,
            alert_type="anomaly_detection"
        )
        
        # Trigger additional actions based on severity
        if severity == AlertLevel.CRITICAL:
            trigger_incident_response(model_id, anomaly_result)
```

### Alert Levels

The system defines four alert levels based on the severity of detected anomalies:

1. **INFO**: Minor anomalies, no immediate action required
2. **WARNING**: Significant anomalies, investigation recommended
3. **ERROR**: Severe anomalies, immediate investigation required
4. **CRITICAL**: Critical anomalies, automated mitigation may be triggered

## Best Practices

### Tuning Anomaly Detection Parameters

For optimal anomaly detection:

1. **Window Size Selection**:
   - For daily metrics: 7-14 days (capture weekly patterns)
   - For hourly metrics: 24-48 hours (capture daily patterns)
   - For high-frequency metrics: 100-1000 data points

2. **Sensitivity Adjustment**:
   - Start with 2.0-3.0 for critical metrics
   - Use 1.5-2.0 for early warning systems
   - Consider increasing threshold for noisy metrics

3. **Method Selection**:
   - Z-score: Good for normally distributed metrics
   - MAD: Better for metrics with outliers
   - ML-based: For complex patterns with sufficient training data

### Avoiding False Positives

Strategies to minimize false alarms:

1. **Smoothing Techniques**:
   - Apply exponential smoothing to reduce noise
   - Use moving averages for trend-based metrics

2. **Confirmation Policies**:
   - Require multiple consecutive anomalies before alerting
   - Combine multiple detection methods

3. **Contextual Awareness**:
   - Incorporate known business events (sales, promotions)
   - Adjust for seasonality (daily, weekly, monthly patterns)

### Operational Considerations

1. **Resource Management**:
   - Batch anomaly detection for efficiency
   - Use incremental algorithms for streaming data

2. **Monitoring the Monitors**:
   - Track false positive/negative rates
   - Periodically review and update thresholds

3. **Alert Management**:
   - Implement alert grouping to prevent alert storms
   - Use progressive escalation for persistent anomalies

## Future Improvements

Planned enhancements to the anomaly detection system:

1. **Deep Learning Integration**:
   - LSTM-based anomaly detection for temporal patterns
   - Autoencoder models for complex metrics

2. **Multivariate Anomaly Detection**:
   - Detect anomalies across correlated metrics
   - Use dimensionality reduction for high-dimensional metrics

3. **Adaptive Thresholds**:
   - Automatically adjust sensitivity based on false alarm rates
   - Learn normal variation patterns from historical data

4. **Explainable Anomalies**:
   - Provide context and potential causes for detected anomalies
   - Link anomalies to potential root causes

## References

1. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. ACM computing surveys (CSUR), 41(3), 1-58.
2. Aggarwal, C. C. (2017). Outlier analysis. Springer International Publishing.
3. Gupta, M., Gao, J., Aggarwal, C. C., & Han, J. (2014). Outlier detection for temporal data: A survey. IEEE Transactions on Knowledge and Data Engineering, 26(9), 2250-2267.

---

*This document was completed on March 11, 2025. It complies with WITHIN ML documentation standards.* 