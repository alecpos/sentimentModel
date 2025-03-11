# Drift Detection Implementation Guide

**IMPLEMENTATION STATUS: IMPLEMENTED**


## Introduction

This guide provides practical implementation details for using the drift detection system in the WITHIN Ad Score & Account Health Predictor. Drift detection is essential for maintaining model reliability in production by identifying when the statistical properties of data or model behavior change over time.

## Table of Contents

1. [Basic Drift Detection](#basic-drift-detection)
2. [Seasonal Drift Handling](#seasonal-drift-handling)
3. [Multivariate Drift Detection](#multivariate-drift-detection)
4. [Advanced Drift Tracking and Forecasting](#advanced-drift-tracking-and-forecasting)
5. [Feature-Level Drift Analysis](#feature-level-drift-analysis)
6. [Model Health Scoring](#model-health-scoring)
7. [Auto-Remediation Integration](#auto-remediation-integration)
8. [Tracking Drift Metrics](#tracking-drift-metrics)
9. [Troubleshooting](#troubleshooting)
10. [Conclusion](#conclusion)

## Basic Drift Detection

### Setting Up Drift Monitoring

```python
from app.services.monitoring.drift_monitoring_service import DriftMonitoringService, DriftMonitoringConfig

# Initialize the service
monitoring_service = DriftMonitoringService(
    storage_path="/path/to/drift/history"
)

# Configure monitoring for a model
config = DriftMonitoringConfig(
    model_id="ad_conversion_predictor_v1",
    check_interval_minutes=60,
    data_drift_threshold=0.05,
    concept_drift_threshold=0.1,
    prediction_drift_threshold=0.05,
    categorical_features=["campaign_type", "device_type", "country"],
    numerical_features=["bid_amount", "daily_budget", "ctr_history"]
)

# Register the model for monitoring
monitoring_service.register_model(config)

# Initialize with reference data
monitoring_service.initialize_reference_data(
    model_id="ad_conversion_predictor_v1",
    reference_data=reference_df,
    reference_predictions=reference_predictions,
    reference_targets=reference_targets
)
```

### Checking for Drift

```python
# Check for drift with new data
drift_result = monitoring_service.check_for_drift(
    model_id="ad_conversion_predictor_v1",
    current_data=current_df,
    current_predictions=current_predictions,
    current_targets=current_targets
)

# Check if drift was detected
if drift_result["drift_detected"]:
    print(f"Drift detected: {drift_result['drift_types']}")
    
    # Get details for specific drift types
    if "data_drift" in drift_result:
        data_drift_details = drift_result["data_drift"]
        print(f"Data drift score: {data_drift_details.get('drift_score')}")
        
    if "concept_drift" in drift_result:
        concept_drift_details = drift_result["concept_drift"]
        print(f"Concept drift detected: {concept_drift_details.get('concept_drift_detected')}")
```

## Seasonal Drift Handling

Seasonal drift handling allows you to maintain separate reference distributions for different time periods or contexts, such as weekdays vs. weekends, holidays vs. regular days, or different seasons.

### Setting Up Seasonal Reference Data

```python
from app.models.ml.monitoring.concept_drift_detector import ConceptDriftDetector

# Initialize the detector
detector = ConceptDriftDetector()

# Initialize reference data for different seasons
detector.initialize_seasonal_reference(
    season_id="weekday",
    X=weekday_features,
    y=weekday_targets
)

detector.initialize_seasonal_reference(
    season_id="weekend",
    X=weekend_features,
    y=weekend_targets
)

# Update predictions for seasonal reference data
detector.update_seasonal_predictions(
    season_id="weekday",
    predictions=weekday_predictions
)

detector.update_seasonal_predictions(
    season_id="weekend",
    predictions=weekend_predictions
)
```

### Checking for Seasonal Drift

```python
# Determine the current season
current_day = datetime.now().strftime("%A")
season_id = "weekend" if current_day in ["Saturday", "Sunday"] else "weekday"

# Check for drift against the appropriate seasonal reference
drift_result = detector.detect_seasonal_drift(
    season_id=season_id,
    current_data=current_features,
    current_predictions=current_predictions,
    current_targets=current_targets
)

if drift_result["drift_detected"]:
    print(f"Seasonal drift detected for {season_id}")
    # Take appropriate action
```

## Multivariate Drift Detection

Multivariate drift detection examines relationships between features, not just individual feature distributions. This is important for capturing complex drift patterns that might not be visible when looking at features in isolation.

### Detecting Multivariate Drift

```python
from app.models.ml.monitoring.drift_detector import DriftDetector

# Initialize detector with multivariate drift detection enabled
detector = DriftDetector(
    categorical_features=["campaign_type", "device_type"],
    numerical_features=["bid_amount", "daily_budget", "ctr_history"],
    detect_multivariate_drift=True
)

# Fit with reference data
detector.fit(reference_data)

# Detect drift including multivariate analysis
drift_result = detector.detect_drift(current_data, multivariate=True)

# Check multivariate drift results
if drift_result.get("multivariate_drift_detected", False):
    print("Multivariate drift detected")
    
    # Check correlation drift
    if drift_result.get("correlation_drift_detected", False):
        print(f"Correlation drift score: {drift_result.get('correlation_drift_score')}")
        
        # Examine specific correlation changes
        correlation_changes = drift_result.get("correlation_changes", {})
        for feature_pair, change in correlation_changes.items():
            print(f"Correlation change in {feature_pair}: {change['absolute_change']}")
    
    # Check joint distribution drift in feature groups
    feature_groups = drift_result.get("feature_groups", {})
    for group_name, group_result in feature_groups.items():
        if group_result.get("drift_detected", False):
            print(f"Joint distribution drift in {group_name}: {group_result.get('features')}")
```

## Advanced Drift Tracking and Forecasting

Advanced drift tracking and forecasting helps you understand drift patterns over time and predict when drift is likely to occur in the future.

### Analyzing Drift Trends

```python
# Analyze drift trends over time
trend_analysis = monitoring_service.analyze_drift_trends(
    model_id="ad_conversion_predictor_v1",
    window_size=10,
    min_events=5
)

print(f"Current drift frequency: {trend_analysis['drift_frequency']}")
print(f"Drift trend direction: {trend_analysis['drift_frequency_trend']}")
print(f"Average interval between drift events: {trend_analysis['avg_interval_hours']} hours")

# Distribution of drift types
drift_types = trend_analysis['drift_types_distribution']
for drift_type, count in drift_types.items():
    print(f"{drift_type}: {count} occurrences")
```

### Forecasting Future Drift

```python
# Forecast drift probability for the next 7 days
forecast = monitoring_service.forecast_drift(
    model_id="ad_conversion_predictor_v1",
    forecast_horizon=7,  # days
    confidence_level=0.9
)

print(f"Next drift expected on: {forecast['next_drift_expected']}")

# Print daily drift probability forecast
print("Drift probability forecast:")
for day in forecast['forecast']:
    print(f"{day['date']}: {day['drift_probability']:.2f} (range: {day['lower_bound']:.2f}-{day['upper_bound']:.2f})")
```

## Feature-Level Drift Analysis

Feature-level drift analysis helps identify which specific features are contributing most to detected drift, allowing for targeted investigation and remediation.

### Analyzing Feature-Level Drift

```python
# Analyze which features contribute most to drift
feature_analysis = monitoring_service.analyze_feature_level_drift(
    model_id="ad_conversion_predictor_v1",
    current_data=current_data,
    top_n=5
)

# Print top drifting features
print("Top drifting features:")
for feature in feature_analysis['top_drifting_features']:
    print(f"{feature['feature']}: drift score = {feature['drift_score']:.4f}")

# Print features with highest impact (drift score * feature importance)
print("\nTop impact features:")
for feature in feature_analysis['top_impact_features']:
    print(f"{feature['feature']}: impact score = {feature['impact_score']:.4f}")

# Examine distribution shifts in numerical features
distribution_shifts = feature_analysis['distribution_shifts']
for feature, shift in distribution_shifts.items():
    print(f"\nDistribution shift in {feature}:")
    print(f"Mean shift: {shift['shifts']['mean_shift']:.4f}")
    print(f"Std deviation shift: {shift['shifts']['std_shift']:.4f}")
    print(f"Range shift: {shift['shifts']['range_shift']:.4f}")
```

## Model Health Scoring

Model health scoring provides a comprehensive assessment of model health based on drift detection, data quality, and performance metrics.

### Calculating Model Health Score

```python
# Calculate overall model health score
health_result = monitoring_service.calculate_model_health_score(
    model_id="ad_conversion_predictor_v1",
    current_data=current_data,
    current_predictions=current_predictions,
    current_targets=current_targets
)

print(f"Overall health score: {health_result['health_score']:.2f}")
print(f"Health status: {health_result['health_status']}")

# Examine individual health components
components = health_result['components']
for component, score in components.items():
    print(f"{component}: {score:.2f}")

# Get recommendations for improving model health
print("\nRecommendations:")
for recommendation in health_result['recommendations']:
    print(f"- {recommendation}")
```

## Auto-Remediation Integration

Auto-remediation allows you to configure automatic responses to detected drift, such as model retraining, rollback to previous versions, or threshold adjustments.

### Configuring Auto-Remediation

```python
from app.services.monitoring.production_monitoring_service import ProductionMonitoringService

# Initialize the production monitoring service
prod_monitoring = ProductionMonitoringService()

# Configure auto-remediation for a model
remediation_config = {
    "enabled_issues": ["data_drift", "concept_drift", "prediction_drift"],
    "severity_threshold": "error",  # Only remediate error or critical issues
    "actions": {
        "data_drift": {
            "type": "retrain",
            "parameters": {
                "dataset": "latest",
                "notify_on_completion": True
            }
        },
        "concept_drift": {
            "type": "rollback",
            "target_version": "v1.2.3"
        },
        "prediction_drift": {
            "type": "adjust_threshold",
            "threshold_delta": 0.05,
            "min_threshold": 0.1,
            "max_threshold": 0.9
        }
    }
}

# Apply the configuration
prod_monitoring.configure_auto_remediation(
    model_id="ad_conversion_predictor_v1",
    remediation_config=remediation_config
)
```

### Executing Remediation Actions

```python
# Execute a remediation action manually
result = prod_monitoring.execute_remediation_action(
    model_id="ad_conversion_predictor_v1",
    issue_type="data_drift",
    severity="error",
    details={
        "drift_score": 0.8,
        "direction": "increase"
    }
)

print(f"Remediation action: {result['action_type']}")
print(f"Success: {result['success']}")
print(f"Message: {result['message']}")

# View remediation history
history = prod_monitoring.get_remediation_history(
    model_id="ad_conversion_predictor_v1",
    issue_types=["data_drift"],
    action_types=["retrain"]
)

print(f"Total remediation actions: {history['total_actions']}")
for action in history['actions']:
    print(f"{action['timestamp']}: {action['action_type']} for {action['issue_type']} - {action['message']}")
```

## Tracking Drift Metrics

```python
# Get drift history for a model
history = monitoring_service.get_drift_history(
    model_id="ad_conversion_predictor_v1",
    start_time=datetime.now() - timedelta(days=30),
    drift_types=["data_drift", "concept_drift"]
)

print(f"Total drift events: {history['total_events']}")

# Create a dashboard for visualizing drift metrics
def create_drift_dashboard(model_id, monitoring_service):
    """Create a dashboard for visualizing drift metrics."""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Get drift history
    history = monitoring_service.get_drift_history(model_id=model_id)
    events = history['drift_events']
    
    if not events:
        print("No drift events to visualize")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'timestamp': datetime.fromisoformat(e['timestamp']),
            'drift_detected': e.get('drift_detected', False),
            'data_drift': 'data_drift' in e.get('drift_types', []),
            'concept_drift': 'concept_drift' in e.get('drift_types', []),
            'prediction_drift': 'prediction_drift' in e.get('drift_types', [])
        }
        for e in events
    ])
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Drift events over time
    df.set_index('timestamp').resample('D').sum().plot(
        y=['data_drift', 'concept_drift', 'prediction_drift'],
        kind='bar',
        stacked=True,
        ax=axes[0],
        title=f'Daily Drift Events for {model_id}'
    )
    
    # Plot 2: Drift frequency over time
    df.set_index('timestamp').resample('W')['drift_detected'].mean().plot(
        ax=axes[1],
        title=f'Weekly Drift Frequency for {model_id}'
    )
    axes[1].set_ylabel('Drift Frequency')
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"{model_id}_drift_dashboard.png")
    print(f"Dashboard saved as {model_id}_drift_dashboard.png")

# Example usage
create_drift_dashboard("ad_conversion_predictor_v1", monitoring_service)
```

## Troubleshooting

### Common Issues

1. **False Positives**: Too many drift alerts
   - Solution: Adjust drift thresholds or use more robust statistical tests
   
2. **False Negatives**: Missing actual drift
   - Solution: Lower thresholds or implement multiple detection methods

3. **High Resource Usage**:
   - Solution: Implement sampling for large datasets or optimize reference data storage

4. **Performance Impact**:
   - Solution: Run drift detection in a separate process/thread or reduce frequency

### Debugging Drift Detection

```python
def debug_drift_detection(model_id, data, reference_data=None):
    """Debug drift detection for a specific model and dataset."""
    
    # Get the detector
    if model_id not in monitoring_service.drift_detectors:
        print(f"No detector found for model {model_id}")
        return
        
    data_drift_detector = monitoring_service.drift_detectors[model_id].get('data_drift')
    if not data_drift_detector:
        print(f"No data drift detector found for model {model_id}")
        return
    
    # If reference data provided, fit with it first
    if reference_data is not None:
        print(f"Fitting detector with provided reference data (n={len(reference_data)})")
        data_drift_detector.fit(reference_data)
    elif not hasattr(data_drift_detector, 'is_fitted') or not data_drift_detector.is_fitted:
        print("Warning: Detector not fitted with reference data")
        return
    
    # Step 1: Check basic properties
    print(f"\n--- Detector Configuration ---")
    print(f"Drift threshold: {data_drift_detector.drift_threshold}")
    print(f"Categorical features: {data_drift_detector.categorical_features}")
    print(f"Numerical features: {data_drift_detector.numerical_features}")
    
    # Step 2: Verbose drift detection
    print(f"\n--- Performing Drift Detection ---")
    drift_result = data_drift_detector.detect_drift(data)
    
    print(f"Overall drift detected: {drift_result.get('drift_detected', False)}")
    
    # Step 3: Feature-level analysis
    print(f"\n--- Feature-Level Drift Analysis ---")
    feature_scores = drift_result.get('feature_drift_scores', {})
    
    if feature_scores:
        drifting_features = []
        for feature, score in feature_scores.items():
            drift_status = "DRIFT" if score > data_drift_detector.drift_threshold else "OK"
            print(f"{feature}: {score:.4f} [{drift_status}]")
            
            if score > data_drift_detector.drift_threshold:
                drifting_features.append(feature)
                
        print(f"\nTop drifting features: {', '.join(drifting_features) if drifting_features else 'None'}")
    else:
        print("No feature-level drift scores available")
    
    # Step 4: Statistical test details
    print(f"\n--- Statistical Test Details ---")
    if 'statistical_tests' in drift_result:
        for test_name, test_result in drift_result['statistical_tests'].items():
            print(f"{test_name}: {test_result}")
    else:
        print("No statistical test details available")
    
    return drift_result
```

### Performance Optimization Tips

1. Use sampling for large datasets:

```python
def detect_drift_with_sampling(model_id, data, sample_size=10000):
    """Detect drift using sampling for large datasets."""
    
    if len(data) > sample_size:
        # Sample data randomly
        sampled_data = data.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} from {len(data)} records")
    else:
        sampled_data = data
        
    return monitoring_service.monitor_model_drift(
        model_id=model_id,
        current_data=sampled_data
    )
```

2. Implement incremental drift detection:

```python
def incremental_drift_detection(model_id, data_stream, batch_size=1000):
    """Process data in batches for drift detection."""
    
    results = []
    for i, batch in enumerate(data_stream.iter_batches(batch_size)):
        print(f"Processing batch {i+1}...")
        
        # Convert batch to DataFrame
        batch_df = pd.DataFrame(batch)
        
        # Check for drift on this batch
        batch_result = monitoring_service.monitor_model_drift(
            model_id=model_id,
            current_data=batch_df
        )
        
        results.append(batch_result)
        
        # If significant drift detected, break early
        if batch_result.get('drift_detected', False) and batch_result.get('severity') in ['error', 'critical']:
            print(f"Significant drift detected in batch {i+1}, stopping...")
            break
    
    # Aggregate results
    overall_drift = any(r.get('drift_detected', False) for r in results)
    drift_types = set()
    for r in results:
        drift_types.update(r.get('drift_types', []))
    
    return {
        'drift_detected': overall_drift,
        'drift_types': list(drift_types),
        'batch_results': results,
        'batches_processed': len(results)
    }
```
  
## Conclusion

This implementation guide provides practical examples for using the drift detection system in the WITHIN Ad Score & Account Health Predictor. By implementing these patterns, you can effectively monitor and respond to drift in your machine learning models, ensuring they maintain their performance and reliability over time.

Key takeaways:

1. **Comprehensive Monitoring**: Monitor for data drift, concept drift, and prediction drift to catch all types of model degradation.
2. **Seasonal Awareness**: Use seasonal reference data to account for expected variations in data patterns.
3. **Multivariate Analysis**: Look beyond individual features to detect complex drift patterns in feature relationships.
4. **Proactive Forecasting**: Analyze drift trends and forecast future drift to take preventive action.
5. **Targeted Investigation**: Use feature-level drift analysis to identify the root causes of drift.
6. **Holistic Health Assessment**: Calculate model health scores to get a comprehensive view of model reliability.
7. **Automated Remediation**: Configure auto-remediation to respond quickly to detected drift.

By following these practices, you can build a robust drift detection system that helps maintain the reliability and performance of your machine learning models in production. 