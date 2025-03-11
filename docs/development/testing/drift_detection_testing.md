# Drift Detection Testing Guide

**IMPLEMENTATION STATUS: IMPLEMENTED**


## Overview

This guide provides a comprehensive approach to testing drift detection systems in the WITHIN Ad Score & Account Health Predictor. It outlines the test-driven development methodology, test categories, approaches for synthetic data generation, and best practices for validating drift detection functionality.

## Table of Contents

1. [Introduction](#introduction)
2. [Test-Driven Development Approach](#test-driven-development-approach)
3. [Test Categories](#test-categories)
4. [Synthetic Data Generation](#synthetic-data-generation)
5. [Testing Statistical Methods](#testing-statistical-methods)
6. [Integration Testing](#integration-testing)
7. [Performance Testing](#performance-testing)
8. [Regression Testing](#regression-testing)
9. [Best Practices](#best-practices)
10. [Troubleshooting Common Issues](#troubleshooting-common-issues)

## Introduction

Testing drift detection systems is particularly challenging because it involves validating statistical methods, ensuring they correctly identify distribution changes while avoiding false positives. Our approach combines unit testing of individual statistical methods with integration testing of the complete drift detection pipeline.

## Test-Driven Development Approach

The WITHIN drift detection system was developed using a test-driven development (TDD) approach:

1. **Write Tests First**: Each feature begins with tests that define the expected behavior
2. **Implement the Feature**: Code is written to pass the tests
3. **Refactor**: Improve the implementation while ensuring tests continue to pass
4. **Expand Coverage**: Add more test cases for edge conditions and special cases

This approach ensures that:
- The drift detection system behaves as expected in all scenarios
- Changes don't break existing functionality
- Documentation exists in the form of tests
- The system is modular and testable by design

## Test Categories

### Unit Tests

Unit tests verify individual components of the drift detection system:

- **Statistical Method Tests**: Validate each distribution comparison method (KS test, KL divergence, Wasserstein distance, etc.)
- **Feature Type Tests**: Ensure correct handling of different feature types (numerical, categorical)
- **Threshold Tests**: Verify behavior with different threshold settings
- **Configuration Tests**: Check initialization with various parameters

### Functional Tests

Functional tests validate complete drift detection workflows:

- **Basic Drift Detection**: Test detection of simple data shifts
- **Multivariate Drift**: Test detection of complex interactions between features
- **Correlation Drift**: Test detection of changes in feature relationships
- **Data Quality Drift**: Test identification of data quality issues
- **Temporal Patterns**: Test detection of various temporal drift patterns

### Edge Case Tests

Edge case tests ensure the system handles unusual scenarios correctly:

- **Small Sample Sizes**: Test with minimal data (e.g., 5-10 samples)
- **Large Sample Sizes**: Test with substantial data (e.g., 100,000+ samples)
- **Missing Values**: Test with incomplete data
- **Extreme Outliers**: Test with anomalous values
- **No Drift Scenario**: Test with identical distributions (no drift)
- **Extreme Drift**: Test with completely different distributions
- **Mixed Drift**: Test with some features drifting and others stable

## Synthetic Data Generation

### Basic Distribution Shift Generation

```python
def generate_drift_data(n_samples=1000, n_features=10, drift_magnitude=0.5, seed=42):
    """
    Generate synthetic data with controlled drift.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        drift_magnitude: Magnitude of the drift (0-1)
        seed: Random seed for reproducibility
        
    Returns:
        reference_data: DataFrame with reference distribution
        drifted_data: DataFrame with drifted distribution
    """
    np.random.seed(seed)
    
    # Generate reference data
    reference_data = pd.DataFrame(
        np.random.normal(0, 1, size=(n_samples, n_features)),
        columns=[f'feature{i}' for i in range(n_features)]
    )
    
    # Generate drifted data with shifted mean
    drifted_data = pd.DataFrame(
        np.random.normal(drift_magnitude, 1, size=(n_samples, n_features)),
        columns=[f'feature{i}' for i in range(n_features)]
    )
    
    return reference_data, drifted_data
```

### Generating Different Drift Types

```python
def generate_sudden_drift(n_samples=1000, n_features=10, drift_magnitude=0.5):
    """Generate data with sudden drift."""
    np.random.seed(42)
    
    # First half - no drift
    first_half = pd.DataFrame(
        np.random.normal(0, 1, size=(n_samples//2, n_features)),
        columns=[f'feature{i}' for i in range(n_features)]
    )
    
    # Second half - drift
    second_half = pd.DataFrame(
        np.random.normal(drift_magnitude, 1, size=(n_samples//2, n_features)),
        columns=[f'feature{i}' for i in range(n_features)]
    )
    
    # Combine
    return pd.concat([first_half, second_half]).reset_index(drop=True)

def generate_gradual_drift(n_samples=1000, n_features=10, max_drift=0.5):
    """Generate data with gradual drift."""
    np.random.seed(42)
    
    # Create data with gradually increasing mean
    data = []
    for i in range(n_samples):
        drift = (i / n_samples) * max_drift
        data.append(np.random.normal(drift, 1, size=n_features))
    
    return pd.DataFrame(data, columns=[f'feature{i}' for i in range(n_features)])

def generate_recurring_drift(n_samples=1000, n_features=10, period=100, magnitude=0.5):
    """Generate data with recurring/cyclical drift."""
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        # Create sinusoidal drift pattern
        drift = magnitude * np.sin(2 * np.pi * i / period)
        data.append(np.random.normal(drift, 1, size=n_features))
    
    return pd.DataFrame(data, columns=[f'feature{i}' for i in range(n_features)])

def generate_correlation_drift(n_samples=1000, n_features=5, correlation_change=0.5):
    """Generate data with correlation drift."""
    np.random.seed(42)
    
    # Reference data - independent features
    ref_data = pd.DataFrame(
        np.random.normal(0, 1, size=(n_samples, n_features)),
        columns=[f'feature{i}' for i in range(n_features)]
    )
    
    # Drifted data - correlated features
    # Create base features
    drifted_data = pd.DataFrame(
        np.random.normal(0, 1, size=(n_samples, n_features)),
        columns=[f'feature{i}' for i in range(n_features)]
    )
    
    # Add correlation between first two features
    drifted_data['feature1'] = drifted_data['feature0'] * correlation_change + \
                               drifted_data['feature1'] * (1 - correlation_change)
    
    return ref_data, drifted_data
```

### Generating Adversarial Test Data

```python
def generate_adversarial_drift_data(n_samples=1000, n_features=10):
    """
    Generate data specifically designed to test multivariate drift detection.
    
    This creates data where univariate drift is minimal but multivariate
    drift (feature interactions) is significant.
    """
    np.random.seed(42)
    
    # Reference data - independent features
    ref_data = pd.DataFrame(
        np.random.normal(0, 1, size=(n_samples, n_features)),
        columns=[f'feature{i}' for i in range(n_features)]
    )
    
    # Create drifted data - same marginal distributions but different joint distribution
    np.random.seed(43)  # Different seed to ensure different data
    drifted_data = pd.DataFrame(
        np.random.normal(0, 1, size=(n_samples, n_features)),
        columns=[f'feature{i}' for i in range(n_features)]
    )
    
    # Introduce correlation between features without changing their individual distributions
    for i in range(0, n_features, 2):
        if i+1 < n_features:
            # Add correlation between pairs of features
            temp = drifted_data[f'feature{i}'].copy()
            drifted_data[f'feature{i}'] = 0.7 * drifted_data[f'feature{i}'] + 0.3 * drifted_data[f'feature{i+1}'] 
            drifted_data[f'feature{i+1}'] = 0.3 * temp + 0.7 * drifted_data[f'feature{i+1}']
    
    return ref_data, drifted_data
```

## Testing Statistical Methods

### KS Test Validation

```python
def test_ks_test_drift_detection():
    """Test KS test drift detection with controlled drift."""
    # Generate data with known drift
    ref_data, drifted_data = generate_drift_data(
        n_samples=1000, 
        n_features=1,
        drift_magnitude=0.5
    )
    
    # Initialize detector with KS test method
    detector = DriftDetector(
        reference_data=ref_data,
        drift_threshold=0.05,
        drift_detection_method='ks_test'
    )
    
    # Check with no drift (should not detect drift)
    no_drift_result = detector.detect_drift(ref_data)
    assert no_drift_result["drift_detected"] is False
    
    # Check with drift (should detect drift)
    drift_result = detector.detect_drift(drifted_data)
    assert drift_result["drift_detected"] is True
    assert drift_result["method"] == "ks_test"
    assert drift_result["drift_score"] > 0.5  # Should have high drift score
```

### Multiple Method Comparison

```python
def test_distribution_comparison_methods():
    """Test different distribution comparison methods."""
    # Generate data with known drift
    ref_data, drifted_data = generate_drift_data(
        n_samples=1000, 
        n_features=1,
        drift_magnitude=0.8  # Large drift for clear detection
    )
    
    # Initialize detector
    detector = DriftDetector(reference_data=ref_data)
    
    # Test KS test
    ks_result = detector.detect_drift(drifted_data, method="ks_test")
    assert ks_result["drift_detected"] is True
    
    # Test KL divergence
    detector.drift_detection_method = "kl_divergence"
    kl_result = detector.detect_drift(drifted_data, method="kl_divergence")
    assert kl_result["drift_detected"] is True
    
    # Test Wasserstein distance
    detector.drift_detection_method = "wasserstein"
    wasserstein_result = detector.detect_drift(drifted_data, method="wasserstein")
    assert wasserstein_result["drift_detected"] is True
    
    # Compare sensitivity (all should detect this obvious drift)
    assert ks_result["drift_score"] > 0.5
    assert kl_result["drift_score"] > 0.5
    assert wasserstein_result["drift_score"] > 0.5
```

### Adversarial Drift Detection Testing

```python
def test_adversarial_drift_detection():
    """
    Test multivariate vs. univariate drift detection with adversarial data.
    
    This test verifies that multivariate drift detection can catch complex
    distribution changes that univariate methods miss.
    """
    # Generate adversarial data
    ref_data, drifted_data = generate_adversarial_drift_data(n_samples=1000, n_features=10)
    
    # Initialize detector with multivariate detection enabled
    detector = DriftDetector(
        reference_data=ref_data,
        drift_threshold=0.05,
        detect_multivariate_drift=True
    )
    
    # Test univariate detection
    univariate_result = detector.detect_drift(drifted_data, multivariate=False)
    
    # Test multivariate detection
    multivariate_result = detector.detect_drift(drifted_data, multivariate=True)
    
    # Multivariate should be more sensitive to this type of drift
    assert multivariate_result["multivariate_drift_score"] > univariate_result["drift_score"]
    
    # For this specific adversarial case, multivariate should detect drift
    # while univariate might not
    assert multivariate_result["multivariate_drift_detected"] is True
```

## Integration Testing

### End-to-End Drift Detection Pipeline

```python
def test_drift_detection_integration():
    """
    Test the end-to-end drift detection pipeline including monitoring service,
    drift detector, and reporting components.
    """
    from app.services.monitoring.drift_monitoring_service import DriftMonitoringService
    from app.services import reporting
    
    # Generate test data
    reference_data, drifted_data = generate_drift_data(n_samples=1000, n_features=5)
    
    # Initialize monitoring service
    monitoring_service = DriftMonitoringService()
    
    # Register model with reference data
    model_id = "test_model"
    monitoring_service.register_model(
        model_id=model_id,
        reference_data=reference_data,
        check_data_drift=True,
        check_prediction_drift=False
    )
    
    # Monitor batch with drifted data
    monitoring_result = monitoring_service.monitor_batch(
        data=drifted_data,
        model_id=model_id
    )
    
    # Check monitoring result
    assert monitoring_result["drift_detected"] is True
    assert "drifted_features" in monitoring_result
    assert len(monitoring_result["drifted_features"]) > 0
    
    # Check if reporting works
    drift_report = reporting.generate_drift_report(
        model_id=model_id,
        drift_results=monitoring_result
    )
    
    # Verify report content
    assert drift_report["model_id"] == model_id
    assert drift_report["drift_detected"] is True
    assert "recommendations" in drift_report
```

### Alert Integration Testing

```python
def test_drift_alerting():
    """Test drift alerting integration."""
    from app.models.ml.monitoring.drift_detector import DriftDetector
    from app.models.ml.monitoring.alert_manager import send_alert
    
    # Generate data with definite drift
    reference_data, drifted_data = generate_drift_data(
        n_samples=1000, 
        n_features=5,
        drift_magnitude=1.0  # Large drift to ensure detection
    )
    
    # Initialize detector with alerting enabled
    detector = DriftDetector(
        reference_data=reference_data,
        drift_threshold=0.05
    )
    detector.alerting_enabled = True
    detector.alert_threshold = 0.1
    
    # Mock the alert sending function to track if it was called
    alert_sent = False
    
    def mock_send_alert(data, message=None, severity=None, alert_type=None):
        nonlocal alert_sent
        alert_sent = True
        # Verify alert data contains expected fields
        assert data["drift_detected"] is True
        assert "drift_scores" in data
        assert len(data["drifted_features"]) > 0
    
    # Replace actual alert function with mock
    original_send_alert = send_alert
    try:
        # Use mock function
        app.models.ml.monitoring.alert_manager.send_alert = mock_send_alert
        
        # Detect drift (should trigger alert)
        detector.detect_drift(drifted_data)
        
        # Verify alert was sent
        assert alert_sent is True
    finally:
        # Restore original function
        app.models.ml.monitoring.alert_manager.send_alert = original_send_alert
```

## Performance Testing

### Benchmark Different Methods

```python
def benchmark_drift_detection_methods(n_samples=1000, n_features=10, n_runs=10):
    """
    Benchmark performance of different drift detection methods.
    
    Args:
        n_samples: Number of samples in the dataset
        n_features: Number of features
        n_runs: Number of benchmark runs
        
    Returns:
        Dictionary of method benchmark results
    """
    import time
    
    # Generate test data
    reference_data, drifted_data = generate_drift_data(
        n_samples=n_samples, 
        n_features=n_features
    )
    
    # Initialize detector
    detector = DriftDetector(reference_data=reference_data)
    
    # Methods to benchmark
    methods = ["ks_test", "kl_divergence", "wasserstein"]
    
    # Store results
    results = {}
    
    # Benchmark each method
    for method in methods:
        detector.drift_detection_method = method
        
        # Warmup run
        detector.detect_drift(drifted_data, method=method)
        
        # Timed runs
        durations = []
        for _ in range(n_runs):
            start_time = time.time()
            detector.detect_drift(drifted_data, method=method)
            end_time = time.time()
            durations.append(end_time - start_time)
        
        # Calculate statistics
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        results[method] = {
            "avg_duration_ms": avg_duration * 1000,
            "min_duration_ms": min_duration * 1000,
            "max_duration_ms": max_duration * 1000
        }
    
    return results
```

### Scaling Test

```python
def test_drift_detection_scaling():
    """Test how drift detection performance scales with data size and dimensionality."""
    import time
    import matplotlib.pyplot as plt
    
    # Data sizes to test
    sample_sizes = [100, 1000, 10000, 100000]
    feature_counts = [5, 10, 50, 100]
    
    # Store results
    size_results = {}
    dim_results = {}
    
    # Test scaling with sample size
    for n_samples in sample_sizes:
        reference_data, drifted_data = generate_drift_data(
            n_samples=n_samples, 
            n_features=10  # Fixed feature count
        )
        
        detector = DriftDetector(reference_data=reference_data)
        
        start_time = time.time()
        detector.detect_drift(drifted_data)
        duration = time.time() - start_time
        
        size_results[n_samples] = duration
    
    # Test scaling with feature dimensionality
    for n_features in feature_counts:
        reference_data, drifted_data = generate_drift_data(
            n_samples=1000,  # Fixed sample size
            n_features=n_features
        )
        
        detector = DriftDetector(reference_data=reference_data)
        
        start_time = time.time()
        detector.detect_drift(drifted_data)
        duration = time.time() - start_time
        
        dim_results[n_features] = duration
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(sample_sizes, [size_results[s] for s in sample_sizes], marker='o')
    plt.xscale('log')
    plt.xlabel('Sample Size')
    plt.ylabel('Detection Time (s)')
    plt.title('Scaling with Sample Size')
    
    plt.subplot(1, 2, 2)
    plt.plot(feature_counts, [dim_results[f] for f in feature_counts], marker='o')
    plt.xlabel('Feature Count')
    plt.ylabel('Detection Time (s)')
    plt.title('Scaling with Dimensionality')
    
    plt.tight_layout()
    plt.savefig('drift_detection_scaling.png')
    
    return {
        'sample_size_scaling': size_results,
        'feature_count_scaling': dim_results
    }
```

## Regression Testing

### Drift Detection Consistency Test

```python
def test_drift_detection_consistency():
    """
    Test that drift detection results are consistent across versions.
    
    This test verifies that code changes don't fundamentally alter 
    detection behavior for established test cases.
    """
    # Load standard test cases
    test_cases = load_standard_test_cases()
    
    for case in test_cases:
        reference_data = case["reference_data"]
        current_data = case["current_data"]
        expected_result = case["expected_result"]
        
        # Initialize detector with standard configuration
        detector = DriftDetector(
            reference_data=reference_data,
            drift_threshold=case["threshold"],
            drift_detection_method=case["method"]
        )
        
        # Run drift detection
        result = detector.detect_drift(current_data)
        
        # Check consistency with expected results
        assert result["drift_detected"] == expected_result["drift_detected"]
        assert abs(result["drift_score"] - expected_result["drift_score"]) < 0.05
        
        # Check feature-level consistency if expected
        if "drifted_features" in expected_result:
            for feature in expected_result["drifted_features"]:
                assert feature in result["drifted_features"]
```

### A/B Test for Detection Methods

```python
def test_detection_method_comparison():
    """
    Compare different detection methods on a standard dataset suite.
    
    This helps identify which methods are most effective for different
    types of drift and data distributions.
    """
    # Load test suite with various drift patterns
    test_suite = load_drift_test_suite()
    
    methods = ["ks_test", "kl_divergence", "wasserstein"]
    results = {method: {"true_positives": 0, "false_positives": 0, 
                        "true_negatives": 0, "false_negatives": 0} 
              for method in methods}
    
    # Test each method on each test case
    for test_case in test_suite:
        reference_data = test_case["reference_data"]
        current_data = test_case["current_data"]
        ground_truth = test_case["has_drift"]
        
        # Test each method
        for method in methods:
            detector = DriftDetector(
                reference_data=reference_data,
                drift_threshold=0.05,
                drift_detection_method=method
            )
            
            result = detector.detect_drift(current_data)
            
            # Update metrics
            if result["drift_detected"] and ground_truth:
                results[method]["true_positives"] += 1
            elif result["drift_detected"] and not ground_truth:
                results[method]["false_positives"] += 1
            elif not result["drift_detected"] and not ground_truth:
                results[method]["true_negatives"] += 1
            else:  # not detected but should have been
                results[method]["false_negatives"] += 1
    
    # Calculate performance metrics
    for method in methods:
        tp = results[method]["true_positives"]
        fp = results[method]["false_positives"]
        tn = results[method]["true_negatives"]
        fn = results[method]["false_negatives"]
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[method]["precision"] = precision
        results[method]["recall"] = recall
        results[method]["f1_score"] = f1
    
    return results
```

## Best Practices

### Creating Effective Tests

1. **Establish Clear Ground Truth**: For each test, define explicitly whether drift should be detected
2. **Control Randomness**: Use fixed random seeds for reproducibility
3. **Test Diverse Scenarios**: Include different data types, drift patterns, and magnitudes
4. **Parameterize Tests**: Use parameterized tests to run the same logic with different configurations
5. **Include Known Edge Cases**: Explicitly test boundary conditions and extreme scenarios
6. **Separate Unit from Integration Tests**: Maintain clear separation between testing levels
7. **Mock Dependencies**: Use mocks for external dependencies to isolate testing
8. **Document Test Purpose**: Each test should have a clear docstring explaining what it validates
9. **Use Descriptive Assertions**: Provide clear error messages in assertions
10. **Maintain Reference Cases**: Keep a set of standardized test cases for regression testing

### Test Data Management

1. **Synthetic Data Generation**: Use controlled data generation to test specific drift patterns
2. **Real-World Samples**: Include anonymized real-world data samples for realistic testing
3. **Versioning**: Version control test datasets alongside code
4. **Data Inspection**: Include visualization tools to inspect test data distributions
5. **Data Size Control**: Use appropriately sized datasets for different testing scenarios

## Troubleshooting Common Issues

### False Positives in Testing

If drift detection tests report false positives:

1. **Check Thresholds**: The drift threshold may be too sensitive
2. **Verify Random Seeds**: Inconsistent random seeds can cause different test runs
3. **Examine Feature Importance**: Look at which features contribute to false positives
4. **Statistical Significance**: Ensure sample sizes are large enough for reliable testing

### False Negatives in Testing

If drift detection fails to identify drift that should be detected:

1. **Drift Magnitude**: Verify that the generated drift is significant enough
2. **Method Selection**: Some methods are less sensitive to certain drift patterns
3. **Feature Selection**: Ensure the drifting features are included in the analysis
4. **Distribution Overlap**: Check if distributions still have substantial overlap

### Unstable Test Results

If test results are inconsistent across runs:

1. **Remove Randomness**: Ensure all random operations use fixed seeds
2. **Check Edge Conditions**: Verify behavior around threshold boundaries
3. **Increase Sample Sizes**: Larger samples provide more statistical stability
4. **Add Test Repetition**: Run critical tests multiple times and aggregate results 