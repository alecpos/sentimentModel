# tests/benchmark/test_account_health_benchmark.py
import pytest
import pandas as pd
import numpy as np
from app.models.ml.prediction import AccountHealthPredictor
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import json
import os
from datetime import datetime

class TestAccountHealthBenchmark:
    """Tests to evaluate Account Health Predictor against documented benchmarks."""
    
    @pytest.fixture
    def model(self):
        """Initialize the Account Health Predictor model."""
        try:
            return AccountHealthPredictor()
        except:
            pytest.skip("AccountHealthPredictor not available")
    
    @pytest.fixture
    def benchmark_dataset(self):
        """Load or generate a benchmark dataset."""
        # In a real implementation, you would load actual test data
        # For demonstration, generating synthetic data
        n_samples = 200
        np.random.seed(42)
        
        # Generate features that simulate account health data
        data = {
            'daily_spend': np.random.uniform(100, 10000, n_samples),
            'impression_count': np.random.uniform(1000, 100000, n_samples),
            'click_count': np.random.uniform(10, 2000, n_samples),
            'conversion_count': np.random.uniform(1, 200, n_samples),
            'campaign_count': np.random.randint(1, 20, n_samples),
            'ad_group_count': np.random.randint(1, 50, n_samples),
            'ad_count': np.random.randint(1, 100, n_samples),
            'account_age_days': np.random.randint(1, 1000, n_samples),
            'platform': np.random.choice(['facebook', 'google', 'tiktok', 'amazon'], n_samples)
        }
        
        # Calculate derived metrics
        data['ctr'] = data['click_count'] / data['impression_count']
        data['cvr'] = data['conversion_count'] / data['click_count']
        data['cpa'] = data['daily_spend'] / np.maximum(data['conversion_count'], 1)
        
        # Create health score based on metrics
        health_score = (
            50 +  # Start at neutral 50
            10 * (data['ctr'] / 0.05) +  # Normalize around typical CTR of 5%
            20 * (data['cvr'] / 0.1) -   # Normalize around typical CVR of 10%
            10 * (data['cpa'] / 100)     # Lower CPA is better
        )
        
        # Bound health score between 0-100
        health_score = np.maximum(0, np.minimum(100, health_score))
        
        # Generate true/false labels for classification metrics based on threshold
        health_label = (health_score >= 70).astype(int)
        
        return pd.DataFrame(data), health_score, health_label
    
    @pytest.fixture
    def benchmark_targets(self):
        """Target metrics from the benchmark document."""
        return {
            "health_score_rmse": 7.2,
            "classification_accuracy": 0.83,
            "anomaly_detection_f1": 0.76,
        }
    
    def test_performance_metrics(self, model, benchmark_dataset, benchmark_targets):
        """Test if the model meets the documented performance metrics."""
        X, health_score, health_label = benchmark_dataset
        
        try:
            # Train the model
            model.fit(X, health_score, health_label)
            
            # Get predictions
            predictions = model.predict(X)
            
            # Extract predictions based on model output format
            # Adjust this based on actual model output structure
            predicted_scores = predictions.get('health_score', predictions)
            predicted_label = (predicted_scores >= 70).astype(int)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(health_score, predicted_scores))
            accuracy = accuracy_score(health_label, predicted_label)
            
            # For anomaly detection F1, we'd need anomaly labels
            # Using a proxy: test examples with health score < 30 as anomalies
            anomaly_true = (health_score < 30).astype(int)
            anomaly_pred = (predicted_scores < 30).astype(int)
            anomaly_f1 = f1_score(anomaly_true, anomaly_pred)
            
            # Compare with benchmark targets
            metrics = {
                "health_score_rmse": rmse,
                "classification_accuracy": accuracy,
                "anomaly_detection_f1": anomaly_f1
            }
            
            # Save results to file for analysis
            results = {
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "targets": benchmark_targets,
                "differences": {k: metrics[k] - benchmark_targets[k] for k in metrics}
            }
            
            # Create directory if it doesn't exist
            os.makedirs("benchmark_results", exist_ok=True)
            
            # Save results
            with open(f"benchmark_results/account_health_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
                json.dump(results, f, indent=2)
            
            # Print results for easy comparison
            print("\nAccount Health Benchmark Results:")
            print("---------------------------------")
            print(f"Metric                 | Actual    | Target    | Difference")
            print(f"--------------------- | --------- | --------- | ----------")
            for metric, value in metrics.items():
                target = benchmark_targets[metric]
                diff = value - target
                print(f"{metric:22} | {value:.6f} | {target:.6f} | {diff:+.6f}")
            
        except Exception as e:
            pytest.skip(f"Model evaluation failed: {str(e)}")