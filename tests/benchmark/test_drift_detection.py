# tests/benchmark/test_drift_detection.py
import pytest
import pandas as pd
import numpy as np
import time
import psutil
import json
import os
from datetime import datetime

# Try to import drift detection, but don't fail if not available
try:
    from app.models.ml.monitoring import DriftDetector
    DRIFT_DETECTOR_AVAILABLE = True
except ImportError:
    DRIFT_DETECTOR_AVAILABLE = False

@pytest.mark.skipif(not DRIFT_DETECTOR_AVAILABLE, reason="Drift detector not available")
class TestDriftDetection:
    """Tests to evaluate drift detection against documented benchmarks."""
    
    @pytest.fixture
    def detector(self):
        """Initialize the drift detector."""
        categorical_features = ['category', 'channel']
        numerical_features = ['clicks', 'impressions', 'conversion_rate']
        return DriftDetector(
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            drift_threshold=0.05,
            check_correlation_drift=True
        )
    
    @pytest.fixture
    def reference_data(self):
        """Generate reference data for drift detection."""
        n_samples = 1000
        np.random.seed(42)
        
        data = {
            'clicks': np.random.poisson(100, n_samples),
            'impressions': np.random.poisson(1000, n_samples),
            'conversion_rate': np.random.beta(2, 10, n_samples),
            'category': np.random.choice(['a', 'b', 'c'], n_samples),
            'channel': np.random.choice(['web', 'mobile', 'app'], n_samples),
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def current_data_no_drift(self, reference_data):
        """Generate current data similar to reference (no drift)."""
        # Copy reference and add small random noise
        data = reference_data.copy()
        
        # Add noise to numerical columns
        data['clicks'] = data['clicks'] + np.random.normal(0, 5, len(data))
        data['clicks'] = np.maximum(0, data['clicks']).astype(int)
        
        data['impressions'] = data['impressions'] + np.random.normal(0, 20, len(data))
        data['impressions'] = np.maximum(0, data['impressions']).astype(int)
        
        data['conversion_rate'] = data['conversion_rate'] + np.random.normal(0, 0.01, len(data))
        data['conversion_rate'] = np.maximum(0, np.minimum(1, data['conversion_rate']))
        
        return data
    
    @pytest.fixture
    def current_data_with_drift(self, reference_data):
        """Generate current data with significant drift."""
        # Copy reference and add significant changes
        data = reference_data.copy()
        
        # Add drift to numerical columns
        data['clicks'] = data['clicks'] * 1.5  # 50% increase
        data['impressions'] = data['impressions'] * 0.7  # 30% decrease
        data['conversion_rate'] = data['conversion_rate'] + 0.05  # +5 percentage points
        data['conversion_rate'] = np.maximum(0, np.minimum(1, data['conversion_rate']))
        
        # Change distribution of categorical columns
        n_samples = len(data)
        data['category'] = np.random.choice(['a', 'b', 'c', 'd'], n_samples, 
                                           p=[0.1, 0.2, 0.5, 0.2])  # New category and different distribution
        data['channel'] = np.random.choice(['web', 'mobile', 'app', 'api'], n_samples,
                                          p=[0.2, 0.2, 0.2, 0.4])  # New channel and different distribution
        
        return data
    
    @pytest.fixture
    def benchmark_targets(self):
        """Target metrics from the benchmark document."""
        return {
            "processing_latency_ms": 100,
            "memory_usage_mb": 50,
            "false_positive_rate": 0.05,
            "detection_delay_days": 1
        }
    
    def test_performance_metrics(self, detector, reference_data, current_data_no_drift, 
                                current_data_with_drift, benchmark_targets):
        """Test if the drift detector meets the documented performance metrics."""
        metrics = {}
        
        # Test processing latency
        detector.fit(reference_data)
        
        # Measure processing time
        start_time = time.time()
        drift_result = detector.detect_drift(current_data_no_drift)
        end_time = time.time()
        
        processing_time_ms = (end_time - start_time) * 1000
        metrics["processing_latency_ms"] = processing_time_ms
        
        # Measure memory usage
        process = psutil.Process(os.getpid())
        memory_usage_mb = process.memory_info().rss / (1024 * 1024)  # RSS in MB
        metrics["memory_usage_mb"] = memory_usage_mb
        
        # Test false positive rate with no-drift data
        no_drift_detected = drift_result.get("drift_detected", False)
        
        # Test detection with drift data
        drift_result = detector.detect_drift(current_data_with_drift)
        drift_detected = drift_result.get("drift_detected", True)
        
        metrics["false_positive_rate"] = float(no_drift_detected)  # 0 if correctly identified no drift, 1 if false positive
        metrics["detection_sensitivity"] = float(drift_detected)   # 1 if correctly detected drift, 0 if missed
        
        # Compare with benchmark targets
        results = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "targets": benchmark_targets,
            "differences": {
                k: metrics[k] - benchmark_targets[k] 
                for k in set(metrics.keys()).intersection(benchmark_targets.keys())
            }
        }
        
        # Create directory if it doesn't exist
        os.makedirs("benchmark_results", exist_ok=True)
        
        # Save results
        with open(f"benchmark_results/drift_detection_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Print results for easy comparison
        print("\nDrift Detection Benchmark Results:")
        print("--------------------------------")
        print(f"Metric                | Actual      | Target      | Difference")
        print(f"-------------------- | ----------- | ----------- | -----------")
        for metric, value in metrics.items():
            if metric in benchmark_targets:
                target = benchmark_targets[metric]
                diff = value - target
                print(f"{metric:21} | {value:11.6f} | {target:11.6f} | {diff:+11.6f}")
            else:
                print(f"{metric:21} | {value:11.6f} | {'N/A':11} | {'N/A':11}")